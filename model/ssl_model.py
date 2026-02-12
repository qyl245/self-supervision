"""Phase 1 self-supervised model: MAE + contrastive + order prediction."""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import info_nce_loss


class MAEDecoder(nn.Module):
    """从序列嵌入重建原始信号"""

    def __init__(self, output_dim: int = 1, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, output_dim, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, seq_emb: torch.Tensor) -> torch.Tensor:
        """seq_emb: (B, S, D) → (B, output_dim, T_recon)"""
        return self.decoder(seq_emb.permute(0, 2, 1))


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, proj_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class OrderPredictionHead(nn.Module):
    """多分类排列预测（num_chunks! 类）"""

    def __init__(self, embed_dim: int = 128, num_chunks: int = 4, dropout: float = 0.1):
        super().__init__()
        num_perms = math.factorial(num_chunks)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, batch_first=True,
            dropout=dropout, dim_feedforward=embed_dim * 2,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_perms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D) → logits (B, num_perms)"""
        return self.classifier(self.transformer(x).mean(dim=1))


# ================================================================
# SSLModel
# ================================================================
class SSLModel(nn.Module):
    """
    Phase 1 自监督模型（EMG 或 IMU 单模态）。
    encoder 是 HybridTimeEncoder。
    """

    def __init__(self, encoder: nn.Module, config: dict, modality: str = "emg"):
        super().__init__()
        self.encoder = encoder
        self.modality = modality

        mcfg = config["model"][modality]
        tcfg = config["train"]["phase1"]
        D = mcfg["embed_dim"]
        self.signal_dim = mcfg["signal_dim"]
        self.num_channels = mcfg.get("num_channels", 1)
        self.temperature = tcfg["temperature"]

        self.mae_decoder = MAEDecoder(
            output_dim=self.signal_dim, embed_dim=D, dropout=mcfg["dropout"]
        )
        self.proj_head = ProjectionHead(D, D, mcfg["dropout"])
        self.order_head = OrderPredictionHead(D, tcfg["num_chunks"], mcfg["dropout"])

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        p = self.modality

        masked = batch.get(f"{p}_mae")
        original = batch.get(f"original_{p}")
        mae_mask = batch.get("mae_mask")
        views = batch.get(f"{p}_views")
        order_seq = batch.get(f"{p}_order")
        order_label = batch.get("order_label")

        if masked is not None and original is not None and mae_mask is not None:
            losses["mae"] = self._mae_loss(masked, original, mae_mask)
        if views is not None:
            losses["contrastive"] = self._contrastive_loss(views)
        if order_seq is not None and order_label is not None:
            losses["order"] = self._order_loss(order_seq, order_label)

        return losses

    def _mae_loss(self, masked, original, mask):
        if self.modality == "emg":
            B, C, T = masked.shape
            flat = masked.reshape(B * C, 1, T)
            _, seq = self.encoder(flat, return_seq=True)
            recon = self.mae_decoder(seq).squeeze(1)      # (B*C, T')
            recon = recon.view(B, C, -1)
        else:
            B, A, T = masked.shape
            _, seq = self.encoder(masked, return_seq=True)
            recon = self.mae_decoder(seq)                  # (B, A, T')

        # Align temporal length between reconstruction and target.
        T_r = recon.shape[-1]
        L = min(T, T_r)
        recon = recon[..., :L]
        original = original[..., :L]
        mask = mask[..., :L]

        # Time-domain loss only on masked positions.
        mask_exp = mask.unsqueeze(1).expand_as(original)

        if mask_exp.any():
            loss_time = F.mse_loss(recon[mask_exp], original[mask_exp])
        else:
            loss_time = recon.new_tensor(0.0)

        # Frequency-domain loss over full signal.
        # Use float32 because cuFFT half precision has shape constraints.
        loss_freq = F.mse_loss(
            torch.fft.rfft(recon.float(), dim=-1).abs(),
            torch.fft.rfft(original.float(), dim=-1).abs(),
        )
        return loss_time + 0.5 * loss_freq

    def _contrastive_loss(self, views):
        B, V = views.shape[:2]
        if self.modality == "emg":
            C, T = views.shape[2], views.shape[3]
            flat = views.reshape(B * V * C, 1, T)
            cls = self.encoder(flat)                   # (B*V*C, D)
            cls = cls.view(B * V, C, -1).mean(dim=1)  # (B*V, D)
        else:
            A, T = views.shape[2], views.shape[3]
            flat = views.reshape(B * V, A, T)
            cls = self.encoder(flat)                   # (B*V, D)

        proj = self.proj_head(cls).view(B, V, -1)
        return info_nce_loss(proj, self.temperature)

    def _order_loss(self, order_seq, label):
        B, K = order_seq.shape[:2]
        if self.modality == "emg":
            C, cT = order_seq.shape[2], order_seq.shape[3]
            flat = order_seq.reshape(B * K * C, 1, cT)
            cls = self.encoder(flat).view(B * K, C, -1).mean(dim=1)
        else:
            A, cT = order_seq.shape[2], order_seq.shape[3]
            flat = order_seq.reshape(B * K, A, cT)
            cls = self.encoder(flat)

        logits = self.order_head(cls.view(B, K, -1))
        return F.cross_entropy(logits, label)
