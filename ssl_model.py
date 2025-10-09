"""
自监督学习模型定义
包含：编码器（EMG/IMU）、任务头（MAE/Projection/Order Prediction）、损失函数、总模型封装
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import traceback

from logging_utils import setup_logger

logger = setup_logger()


# ================================================================
# 编码器
# ================================================================
class HybridTimeEncoder(nn.Module):
    """
    通用混合编码器(CNN + Transformer)
    支持输入: (B, input_dim, T)
    输出 patch embeddings 或 CLS token embedding
    """

    def __init__(self, input_dim=1, embed_dim=128, nhead=8,
                 num_transformer_layers=4, dropout=0.1,
                 use_sinusoidal_pos=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.pos_norm = nn.LayerNorm(embed_dim)

        # CNN特征提取
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, embed_dim, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_sinusoidal_pos:
            self.register_buffer('pos_encoder_base', self._generate_sinusoidal_pe(embed_dim))
        else:
            self.pos_encoder = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def _generate_sinusoidal_pe(self, embed_dim, max_len=200):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _compute_seq_len(self, x):
        T = x.shape[-1]
        after_first = (T + 8 - 9) // 2 + 1
        after_second = (after_first + 8 - 9) // 4 + 1
        return after_second

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, input_dim, T)
        输出: (B, total_len, embed_dim)  带CLS token的序列
        """
        B, _, T = x.shape
        seq_len = self._compute_seq_len(x)

        x = self.cnn_extractor(x)                     # -> (B, embed_dim, seq_len)
        x = x.permute(0, 2, 1)                        # -> (B, seq_len, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)         # -> (B, seq_len+1, embed_dim)

        total_len = seq_len + 1
        if self.use_sinusoidal_pos:
            pos_enc = self.pos_encoder_base[:, :total_len, :]
        else:
            pos_enc = self.pos_encoder.expand(B, total_len, -1)
        x = x + pos_enc

        x = self.pos_norm(x)
        x = self.transformer_encoder(x)               # -> (B, total_len, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 CLS token embedding 用于对比 / 分类任务
        """
        features = self.forward_features(x)
        return features[:, 0, :]  # (B, embed_dim)


class EMGTimeEncoder(nn.Module):
    """
    EMG数据编码器: (B, num_channels, T) -> (B, num_channels, embed_dim)
    每个通道用相同的 HybridTimeEncoder 处理
    """

    def __init__(self, num_channels=16, embed_dim=128,
                 nhead=8, num_transformer_layers=2, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.encoder_per_channel = HybridTimeEncoder(
            input_dim=1,
            embed_dim=embed_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"[EMG Encoder] Expected shape (B,C,T), got {x.shape}")
        B, C, T = x.shape
        x_reshaped = x.view(B * C, 1, T)                          # Conv1d格式 (B*C, 1, T)
        embedding = self.encoder_per_channel(x_reshaped)          # (B*C, embed_dim)
        return embedding.view(B, C, self.embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class IMUTimeEncoder(nn.Module):
    """
    IMU数据编码器
    支持:
      - 原始4D: (B, num_sensors, num_axes, T)
      - Flatten后的3D: (B, channels_flat, T)
    """

    def __init__(self, num_sensors=8, num_axes=6, embed_dim=128,
                 nhead=8, num_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_axes = num_axes
        self.embed_dim = embed_dim
        self.encoder_per_sensor = HybridTimeEncoder(
            input_dim=num_axes,
            embed_dim=embed_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # 原始格式
            B, C_sensors, A, T = x.shape
            if A != self.num_axes:
                raise ValueError(f"[IMU Encoder] Expected num_axes={self.num_axes}, got {A}")
            x_reshaped = x.view(B * C_sensors, A, T)               # Conv1d格式 (B*num_sensors, num_axes, T)
            embedding = self.encoder_per_sensor(x_reshaped)        # (B*num_sensors, embed_dim)
            return embedding.view(B, C_sensors, self.embed_dim)

        elif x.ndim == 3:
            # Flatten后的格式: (B_total, channels_flat, T)
            B_total, C_flat, T = x.shape
            expected_flat = self.num_sensors * self.num_axes
            if C_flat != expected_flat:
                raise ValueError(f"[IMU Encoder] Expected C_flat={expected_flat} (sensors*axes), got {C_flat}")
        
            # 还原形状: 每个传感器有num_axes轴
            x_reshaped = x.view(B_total * self.num_sensors, self.num_axes, T)
            embedding = self.encoder_per_sensor(x_reshaped)  # -> (B_total*num_sensors, embed_dim)
            return embedding.view(B_total, self.num_sensors, self.embed_dim)
            
        else:
            raise ValueError(f"[IMU Encoder] Unexpected input shape {x.shape}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

        
# ================================================================
# 任务头
# ================================================================
# --- 任务A: MAE 的解码器 ---
class MAEDecoder(nn.Module):
    """
    从Transformer的patch embedding重建原始信号。
    使用转置卷积(ConvTranspose1d)进行上采样。
    """

    def __init__(self, input_dim=1, embed_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            # 第一层：embed_dim -> 64, 上采样4倍
            nn.ConvTranspose1d(embed_dim, 64, kernel_size=9, stride=4, padding=4, output_padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第二层：64 -> input_dim, 上采样2倍
            nn.ConvTranspose1d(64, input_dim, kernel_size=9, stride=2, padding=4, output_padding=1),
        )

        # 可选的最终激活（根据信号范围选择）
        self.final_proj = nn.Identity()  # 或 nn.Tanh() 如果信号归一化到[-1,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches+1, embed_dim), 包含CLS token
        # 移除CLS token，只保留信号patches
        if x.shape[1] > 1:  # 确保有patches
            signal_patches = x[:, 1:, :]  # -> (B, num_patches, embed_dim)
        else:
            raise ValueError("Expected at least 2 tokens (CLS + patches), got shape: {}".format(x.shape))

        # 转置为卷积格式
        signal_patches = signal_patches.permute(0, 2, 1)  # -> (B, embed_dim, num_patches)

        # 通过解码器重建
        reconstructed = self.decoder(signal_patches)  # -> (B, input_dim, T)
        return self.final_proj(reconstructed)


# --- 任务B: 对比学习的投影头 ---
class ProjectionHead(nn.Module):
    """
    用于对比学习的MLP投影头。
    """

    def __init__(self, embed_dim=128, projection_dim=128, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, projection_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# --- 任务C: 时序顺序预测的头 ---
class OrderPredictionHead(nn.Module):
    """
    接收一个窗口嵌入序列，并预测其顺序是否正确。
    """

    def __init__(self, embed_dim=128, num_chunks=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=embed_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)  # 2 classes: ordered vs shuffled
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_chunks, embed_dim)
        features = self.transformer(x)
        # 使用序列的平均特征进行分类
        pooled_features = features.mean(dim=1)  # (B, embed_dim)
        return self.classifier(pooled_features)



# ================================================================
# 损失
# ================================================================
# --- 改进的 InfoNCE 损失函数 ---
def info_nce_loss(features: torch.Tensor, temperature: float = 0.1,
                  labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算InfoNCE损失
    Args:
        features: (B, N, D) 其中N是视图数量
        temperature: 温度参数
        labels: 可选的正样本标签，默认假设第2个视图是正样本
    """
    B, N, D = features.shape
    device = features.device

    if N < 2:
        raise ValueError(f"Need at least 2 views for contrastive learning, got {N}")

    # L2归一化
    features = F.normalize(features, dim=2)

    # 计算相似度矩阵
    # features: (B, N, D) -> (B*N, D)
    features_flat = features.view(B * N, D)
    similarity_matrix = torch.matmul(features_flat, features_flat.T) / temperature  # (B*N, B*N)

    # 创建标签：对于每个anchor，其对应的positive是同一个batch中的另一个view
    if labels is None:
        # 默认策略：view 0和view 1互为正样本对
        labels = torch.arange(B * N, device=device)
        for i in range(B):
            labels[i * N] = i * N + 1  # view 0 -> view 1
            labels[i * N + 1] = i * N  # view 1 -> view 0

    # 移除自相似度（对角线）
    mask = torch.eye(B * N, device=device).bool()
    similarity_matrix.masked_fill_(mask, -float('inf'))

    # 计算交叉熵损失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss



# ================================================================
# 总模型
# ================================================================
class SSL_Model(nn.Module):
    """
    多任务自监督学习模型
    带 debug 开关 — debug=True 时输出详细调试信息
    """

    def __init__(self, time_encoder: nn.Module, config: Dict, debug: bool = False):
        super().__init__()
        self.time_encoder = time_encoder
        self.config = config
        self.debug = debug  # 新增调试开关

        embed_dim = config['embed_dim']
        signal_dim = config['signal_dim']

        self.mae_decoder = MAEDecoder(
            input_dim=signal_dim,
            embed_dim=embed_dim,
            dropout=config.get('dropout', 0.1)
        )
        self.projection_head = ProjectionHead(
            embed_dim=embed_dim,
            projection_dim=config.get('projection_dim', embed_dim),
            dropout=config.get('dropout', 0.1)
        )
        self.order_prediction_head = OrderPredictionHead(
            embed_dim=embed_dim,
            num_chunks=config.get('num_chunks', 3),
            dropout=config.get('dropout', 0.1)
        )

    # ===== 新增统一调试方法 =====
    def _debug(self, msg: str, error: bool = False):
        """打印或静默调试信息"""
        if self.debug:
            if error:
                warnings.warn(msg)
            else:
                print(msg)

    def _ensure_device_consistency(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    def _ensure_tensor(self, x):
        if isinstance(x, (tuple, list)):
            self._debug(f"[DEBUG _ensure_tensor] input is {type(x)}, length={len(x)}")
            return x[0]
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        kwargs = self._ensure_device_consistency(kwargs)
        losses, outputs = {}, {}
        shared_encoded = None

        if self.debug:
            print("[DEBUG forward initial keys/shapes]:")
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  key={k}, shape={tuple(v.shape)}")
                elif isinstance(v, (tuple, list)):
                    print(f"  key={k}, type={type(v)}, len={len(v)}")
                else:
                    print(f"  key={k}, type={type(v)}")

        # === Contrastive Views ===
        if 'contrastive_views' in kwargs:
            views = self._ensure_tensor(kwargs['contrastive_views'])
            if views.ndim != 4:
                raise ValueError(f"[Contrastive Debug] Expected 4D, got shape={tuple(views.shape)}")
            B, N, C, T = views.shape
            self._debug(f"[DEBUG contrastive_views shape unpack] B={B}, N={N}, C={C}, T={T}")
            views_flat = views.view(B * N, C, T)
            shared_encoded = self.time_encoder(views_flat)
            outputs['encoded_views'] = shared_encoded

        # === MAE 任务 ===
        if all(k in kwargs for k in ['mae_input', 'original_signal', 'mask']):
            try:
                masked_input = self._ensure_tensor(kwargs['mae_input'])
                original_signal = self._ensure_tensor(kwargs['original_signal'])
                mask = kwargs['mask']

                if self.debug:
                    print(f"[DEBUG mae_input shape] {tuple(masked_input.shape)}")

                if masked_input.ndim == 3:
                    B, C, T = masked_input.shape
                    input_reshaped = masked_input.view(B * C, 1, T) if self.config['signal_dim'] == 1 else masked_input.view(B * C, self.config['signal_dim'], T)
                elif masked_input.ndim == 4:
                    B, C, Ax, T = masked_input.shape
                    input_reshaped = masked_input.view(B * C, Ax, T)
                else:
                    raise ValueError(f"[MAE Debug] Unexpected mae_input shape={tuple(masked_input.shape)}")

                if hasattr(self.time_encoder, 'forward_features'):
                    all_patch_embeddings = self.time_encoder.forward_features(input_reshaped)
                else:
                    all_patch_embeddings = self.time_encoder(input_reshaped)

                reconstructed_signal = self.mae_decoder(all_patch_embeddings)
                reconstructed_signal = reconstructed_signal.view(B, C, self.config['signal_dim'], T)

                if self.config['signal_dim'] == 1:
                    reconstructed_signal = reconstructed_signal.squeeze(2)
                    mask_expanded = mask.unsqueeze(1).expand(-1, C, -1)
                else:
                    mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(-1, C, self.config['signal_dim'], -1)

                if mask_expanded.sum() > 0:
                    loss_mae = F.mse_loss(
                        reconstructed_signal[mask_expanded],
                        original_signal[mask_expanded]
                    )
                    losses['mae_loss'] = loss_mae
                    outputs['reconstructed_signal'] = reconstructed_signal
                else:
                    self._debug("No masked tokens for MAE", error=True)

            except Exception as e:
                self._debug(f"[MAE Debug] Error: {e}", error=True)

        # === Order Prediction 任务 ===
        if all(k in kwargs for k in ['order_sequence', 'order_label']):
            try:
                sequence = self._ensure_tensor(kwargs['order_sequence'])
                if sequence.ndim != 4:
                    raise ValueError(f"[Order Seq Debug] Expected 4D, got shape={tuple(sequence.shape)}")
                B, N_chunks, C, T = sequence.shape
                seq_flat = sequence.view(B * N_chunks, C, T)
                channel_embeddings = shared_encoded if shared_encoded is not None else self.time_encoder(seq_flat)
                window_embeddings = channel_embeddings.mean(dim=1)
                seq_embeddings = window_embeddings.view(B, N_chunks, -1)
                logits = self.order_prediction_head(seq_embeddings)
                losses['order_loss'] = F.cross_entropy(logits, kwargs['order_label'])
                outputs['order_logits'] = logits
            except Exception as e:
                self._debug(f"[Order Prediction Debug] Error: {e}", error=True)

        # === Contrastive 任务 ===
        if 'contrastive_views' in kwargs:
            try:
                views = self._ensure_tensor(kwargs['contrastive_views'])
                B, N, C, T = views.shape
                views_flat = views.view(B * N, C, T)
                channel_embeddings = shared_encoded if shared_encoded is not None else self.time_encoder(views_flat)
                window_embeddings = channel_embeddings.mean(dim=1)
                projections = self.projection_head(window_embeddings).view(B, N, -1)
                losses['contrastive_loss'] = info_nce_loss(
                    projections,
                    temperature=self.config.get('temperature', 0.1)
                )
                outputs['projections'] = projections
            except Exception as e:
                self._debug(f"[Contrastive Debug] Error: {e}", error=True)

        return {'losses': losses, 'outputs': outputs}


# ================================================================
# 方法
# ================================================================
def create_emg_encoder(config):
    return EMGTimeEncoder(
        num_channels=config['model']['emg']['num_channels'],
        embed_dim=config['model']['emg']['embed_dim'],
        nhead=config['model']['emg']['nhead'],
        num_transformer_layers=config['model']['emg']['num_transformer_layers'],
        dropout=config['model']['emg']['dropout']
    )


def create_imu_encoder(config):
    return IMUTimeEncoder(
        num_sensors=config['model']['imu']['num_sensors'],
        num_axes=config['model']['imu']['num_axes'],
        embed_dim=config['model']['imu']['embed_dim'],
        nhead=config['model']['imu']['nhead'],
        num_transformer_layers=config['model']['imu']['num_transformer_layers'],
        dropout=config['model']['imu']['dropout']
    )


def create_ssl_model(encoder, config, modality='emg'):
    cfg = config['model'][modality]
    cfg.update({
        'embed_dim': cfg['embed_dim'],
        'signal_dim': cfg['signal_dim'],
        'projection_dim': config['train'].get('projection_dim', cfg['embed_dim']),
        'num_chunks': config['train']['num_chunks'],
        'temperature': config['train']['temperature'],
        'dropout': cfg['dropout']
    })

    return SSL_Model(encoder, cfg)