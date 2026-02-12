"""Phase 2 multimodal fusion model."""
import torch
import torch.nn as nn
from typing import Dict, Optional

from models.encoder import EMGEncoder, IMUEncoder


# ================================================================
# SetEncoder — 模态内通道聚合
# ================================================================
class SetEncoder(nn.Module):
    """CLS Token + Transformer 将多通道嵌入聚合为全局表示"""

    def __init__(self, embed_dim: int, nhead: int, num_layers: int,
                 max_tokens: int = 64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, D)
        Returns: global_emb (B, D), context_embs (B, C, D)
        """
        B, C, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)                  # (B, C+1, D)
        h = h + self.pos_embed[:, :C + 1, :]
        h = self.transformer(h)
        return h[:, 0], h[:, 1:]                         # global, context


# ================================================================
# CrossModalFusion — 修复 LayerNorm 共享 bug
# ================================================================
class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention with residual normalization.
    """

    def __init__(self, embed_dim: int, nhead: int):
        super().__init__()
        self.cross_emg2imu = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.cross_imu2emg = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

        self.norm_emg_seq = nn.LayerNorm(embed_dim)
        self.norm_imu_seq = nn.LayerNorm(embed_dim)
        self.norm_emg_glob = nn.LayerNorm(embed_dim)
        self.norm_imu_glob = nn.LayerNorm(embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, emg_ctx, imu_ctx, emg_global, imu_global):
        """
        emg_ctx   (B, C_emg, D)    imu_ctx   (B, S_imu, D)
        emg_global (B, D)           imu_global (B, D)
        Returns: fused (B, D), emg_feat (B, D), imu_feat (B, D)
        """
        # 双向 Cross-Attention
        emg_cross, _ = self.cross_emg2imu(query=emg_ctx, key=imu_ctx, value=imu_ctx)
        imu_cross, _ = self.cross_imu2emg(query=imu_ctx, key=emg_ctx, value=emg_ctx)

        # 序列级残差 + LayerNorm
        emg_enh = self.norm_emg_seq(emg_ctx + emg_cross)
        imu_enh = self.norm_imu_seq(imu_ctx + imu_cross)

        # 聚合 → 全局
        emg_enh_g = emg_enh.mean(dim=1)
        imu_enh_g = imu_enh.mean(dim=1)

        # 全局级残差 + LayerNorm（独立）
        emg_feat = self.norm_emg_glob(emg_global + emg_enh_g)
        imu_feat = self.norm_imu_glob(imu_global + imu_enh_g)

        # MLP 融合
        fused = self.fusion_mlp(torch.cat([emg_feat, imu_feat], dim=-1))
        return fused, emg_feat, imu_feat


# ================================================================
# Phase 2 投影头
# ================================================================
class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in), nn.BatchNorm1d(d_in), nn.ReLU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# MultiModalModel — Phase 2 完整模型
# ================================================================
class MultiModalModel(nn.Module):
    """
    Cross-modal classifier with optional single-modality inference.
    """

    def __init__(self, emg_encoder: EMGEncoder, imu_encoder: IMUEncoder,
                 model_cfg: dict, num_classes: int):
        super().__init__()
        self.emg_encoder = emg_encoder
        self.imu_encoder = imu_encoder

        D = model_cfg["fusion"]["embed_dim"]
        fcfg = model_cfg["fusion"]

        self.emg_set = SetEncoder(
            embed_dim=D, nhead=fcfg["set_encoder_heads"],
            num_layers=fcfg["set_encoder_layers"],
            max_tokens=model_cfg["emg"]["num_channels"],
        )

        self.fusion = CrossModalFusion(D, fcfg["fusion_heads"])
        self.modality_gate = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Linear(D, 2),
        )

        # 投影头（对比学习用）
        self.proj_cross = ProjectionHead(D, D)
        self.proj_emg = ProjectionHead(D, D)
        self.proj_imu = ProjectionHead(D, D)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D, num_classes),
        )
        self.emg_classifier = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D, num_classes),
        )
        self.imu_classifier = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D, num_classes),
        )

    def _encode_emg(self, emg: torch.Tensor):
        emg_ch = self.emg_encoder(emg, return_seq=False)       # (B, 8, D)
        emg_global, emg_ctx = self.emg_set(emg_ch)             # (B, D), (B, 8, D)
        return emg_global, emg_ctx

    def _encode_imu(self, imu: torch.Tensor):
        imu_cls, imu_seq = self.imu_encoder(imu, return_seq=True)  # (B, D), (B, S, D)
        return imu_cls, imu_seq

    def forward(self, emg: torch.Tensor, imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        emg: (B, C, T), imu: (B, A, T)
        """
        # EMG / IMU — 单次前向
        emg_global, emg_ctx = self._encode_emg(emg)
        imu_cls, imu_seq = self._encode_imu(imu)

        # 跨模态融合
        fused, emg_feat, imu_feat = self.fusion(
            emg_ctx, imu_seq, emg_global, imu_cls
        )
        gate_logits = self.modality_gate(torch.cat([emg_feat, imu_feat], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        gated = gate[:, :1] * emg_feat + gate[:, 1:] * imu_feat
        fused_final = 0.5 * (fused + gated)

        return {
            "logits": self.classifier(fused_final),
            "fused_proj": self.proj_cross(fused_final),
            "emg_proj": self.proj_emg(emg_feat),
            "imu_proj": self.proj_imu(imu_feat),
            "modality_gate": gate,
        }

    def forward_emg_only(self, emg: torch.Tensor, imu_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        EMG-only inference path without cross-modal fusion.
        """
        emg_feat, _ = self._encode_emg(emg)
        logits = self.emg_classifier(emg_feat)
        fused_proj = self.proj_cross(emg_feat)
        emg_proj = self.proj_emg(emg_feat)
        if imu_ref is not None:
            B = imu_ref.shape[0]
            D = emg_feat.shape[-1]
            imu_proj = emg_feat.new_zeros(B, D)
        else:
            imu_proj = emg_feat.new_zeros(emg_feat.shape[0], emg_feat.shape[1])
        return {
            "logits": logits,
            "fused_proj": fused_proj,
            "emg_proj": emg_proj,
            "imu_proj": imu_proj,
        }

    def forward_imu_only(self, imu: torch.Tensor, emg_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        IMU-only inference path without cross-modal fusion.
        """
        imu_feat, _ = self._encode_imu(imu)
        logits = self.imu_classifier(imu_feat)
        fused_proj = self.proj_cross(imu_feat)
        imu_proj = self.proj_imu(imu_feat)
        if emg_ref is not None:
            B = emg_ref.shape[0]
            D = imu_feat.shape[-1]
            emg_proj = imu_feat.new_zeros(B, D)
        else:
            emg_proj = imu_feat.new_zeros(imu_feat.shape[0], imu_feat.shape[1])
        return {
            "logits": logits,
            "fused_proj": fused_proj,
            "emg_proj": emg_proj,
            "imu_proj": imu_proj,
        }

    def forward_by_mode(self, emg: torch.Tensor, imu: torch.Tensor, input_mode: str = "both") -> Dict[str, torch.Tensor]:
        if input_mode == "emg_only":
            return self.forward_emg_only(emg, imu_ref=imu)
        if input_mode == "imu_only":
            return self.forward_imu_only(imu, emg_ref=emg)
        return self.forward(emg, imu)
