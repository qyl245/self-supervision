import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import yaml
from ssl_model import create_emg_encoder, create_imu_encoder
from demo2 import build_phase2_loaders
from datasets import KneePAD, MovePort
from demo2 import augment_imu, augment_emg


# ==== Step 1: 加载预训练的 TimeEncoder 并冻结策略 ====
def load_pretrained_encoder(encoder, ckpt_path, freeze_cnn=True):
    print(f"[INFO] Loading pretrained encoder from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state)
    for name, p in encoder.named_parameters():
        if freeze_cnn and "cnn" in name.lower():  # 按名称冻结低层
            p.requires_grad = False
        else:
            p.requires_grad = True
    return encoder


# ==== Step 2: 模态内编码器 ====
class SetEncoder(nn.Module):
    def __init__(self, num_members, embed_dim, nhead, layers):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_members + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        B = x.shape[0]
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1) + self.pos_embed
        x = self.transformer(x)

        global_emb = x[:, 0]
        context_embs = x[:, 1:]
        return global_emb, context_embs


# ==== Step 3: 跨模态残差融合模块 ====
class ResidualFusion(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.cross_emg_to_imu = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.cross_imu_to_emg = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm_emg = nn.LayerNorm(embed_dim)
        self.norm_imu = nn.LayerNorm(embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, emg_embs, imu_embs, emg_global, imu_global):
        emg_cross, _ = self.cross_emg_to_imu(emg_embs, imu_embs, imu_embs)
        imu_cross, _ = self.cross_imu_to_emg(imu_embs, emg_embs, emg_embs)

        emg_enh = self.norm_emg(emg_embs + emg_cross)
        imu_enh = self.norm_imu(imu_embs + imu_cross)

        emg_feat = emg_global + emg_enh.mean(dim=1)
        imu_feat = imu_global + imu_enh.mean(dim=1)

        # 融合前归一化
        emg_feat = self.norm_emg(emg_feat)
        imu_feat = self.norm_imu(imu_feat)

        fused = torch.cat([emg_feat, imu_feat], dim=1)
        fused_out = self.fusion_mlp(fused)
        return fused_out, emg_feat, imu_feat


class TemporalContextModule(nn.Module):
    def __init__(self, embed_dim, nhead=4, layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        # 假设 fused_emb 是(B, D)，扩展成(B, T, D)可以是多个窗口堆叠
        # 此处为了简单直接加一个维度
        x = x.unsqueeze(1)  # (B, 1, D)
        x = self.transformer(x)
        return x.squeeze(1)


# ==== Step 4: 投影头 ====
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==== Step 5: 融合模型 ====
class MultiModalSSL(nn.Module):
    def __init__(self, emg_enc, imu_enc, cfg):
        super().__init__()
        self.emg_enc = emg_enc
        self.imu_enc = imu_enc
        edim = cfg['fusion']['embed_dim']

        self.emg_set = SetEncoder(cfg['emg']['num_channels'], edim,
                                  cfg['fusion']['set_encoder_heads'], cfg['fusion']['set_encoder_layers'])
        self.imu_set = SetEncoder(cfg['imu']['num_sensors'], edim,
                                  cfg['fusion']['set_encoder_heads'], cfg['fusion']['set_encoder_layers'])

        self.fusion = ResidualFusion(edim, cfg['fusion']['fusion_heads'])
        self.tcm = TemporalContextModule(edim, nhead=4, layers=2)  # 新增 TCM

        self.proj_cross = ProjectionHead(edim, edim)  # 跨模态对齐
        self.proj_emg = ProjectionHead(edim, edim)  # 模态内任务
        self.proj_imu = ProjectionHead(edim, edim)

        # 下游线性分类头
        self.ce_head_emg = nn.Linear(edim, cfg['fusion']['num_classes'])
        self.ce_head_imu = nn.Linear(edim, cfg['fusion']['num_classes'])
        self.ce_head_fused = nn.Linear(edim, cfg['fusion']['num_classes'])

    def forward(self, emg_data, imu_data):
        # === Phase1 原始 encoder 全局向量（直通路径） ===
        emg_tokens = self.emg_enc(emg_data, return_seq=False)  # (B, C, D)
        imu_tokens = self.imu_enc(imu_data, return_seq=False)  # (B, S, D)
        emg_raw_global = emg_tokens.mean(dim=1)  # (B, D)
        imu_raw_global = imu_tokens.mean(dim=1)  # (B, D)

        # === Phase2 SetEncoder 表示（用于融合） ===
        emg_ch, _ = self.emg_enc(emg_data, return_seq=True)
        imu_ch, _ = self.imu_enc(imu_data, return_seq=True)
        emg_glob, emg_ctx = self.emg_set(emg_ch)
        imu_glob, imu_ctx = self.imu_set(imu_ch)

        fused_emb, emg_feat, imu_feat = self.fusion(emg_ctx, imu_ctx, emg_glob, imu_glob)
        fused_emb = self.tcm(fused_emb)

        return {
            'fused_emb': self.proj_cross(fused_emb),
            'emg_emb': self.proj_emg(emg_feat),
            'imu_emb': self.proj_imu(imu_feat),
            'emg_cls_raw': emg_raw_global,  # 原生Phase1路径
            'imu_cls_raw': imu_raw_global,
            'fused_cls': fused_emb
        }


# ==== Step 6: LOSS ====
def info_nce(a, b, temp=0.1):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = torch.matmul(a, b.T) / temp
    labels = torch.arange(a.size(0), device=a.device)
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    return (loss_a2b + loss_b2a) / 2


# ==== Step 7: 训练循环 ====
def train_phase2(cfg, train_loader, model, optimizer):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg['epochs']):
        w_cross, w_emg, w_imu = get_loss_weights(cfg, epoch)
        w_ce = cfg.get('w_ce', 0.1)

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")

        for batch in pbar:
            emg = batch['emg'].to(device)
            imu = batch['imu'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None

            # 两个视图做模态内保护
            emg_v1 = augment_emg(emg)
            emg_v2 = augment_emg(emg)
            imu_v1 = augment_imu(imu)
            imu_v2 = augment_imu(imu)

            with torch.cuda.amp.autocast():  # 半精度运行
                out_v1 = model(emg_v1, imu_v1)
                out_v2 = model(emg_v2, imu_v2)

                loss_cross = info_nce(out_v1['fused_emb'], out_v2['fused_emb'])
                loss_emg_intra = info_nce(out_v1['emg_emb'], out_v2['emg_emb'])
                loss_imu_intra = info_nce(out_v1['imu_emb'], out_v2['imu_emb'])

                loss = w_cross * loss_cross + w_emg * loss_emg_intra + w_imu * loss_imu_intra

                # 弱监督（可选）—直接用 raw cls 做分类
                if labels is not None and w_ce > 0:
                    ce_emg = F.cross_entropy(model.ce_head_emg(out_v1['emg_cls_raw']), labels)
                    ce_imu = F.cross_entropy(model.ce_head_imu(out_v1['imu_cls_raw']), labels)
                    ce_fused = F.cross_entropy(model.ce_head_fused(out_v1['fused_cls']), labels)
                    loss += w_ce * (ce_emg + ce_imu + ce_fused)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        print(f"[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'phase2_final_model.pt')


# ==== Step 8: 权重调度 ====
def get_loss_weights(cfg, epoch):
    if cfg.get('schedule_type', 'staged') == 'staged':
        schedule = cfg['loss_schedule']
        for _, stage_cfg in schedule.items():
            start, end = map(int, stage_cfg['epochs'].split('-'))
            if start <= epoch + 1 <= end:
                return stage_cfg['w_cross'], stage_cfg['w_emg'], stage_cfg['w_imu']
        last_stage = list(schedule.values())[-1]
        return last_stage['w_cross'], last_stage['w_emg'], last_stage['w_imu']
    elif cfg['schedule_type'] == 'linear':
        total_epochs = cfg['num_epochs']

        def lerp(s, e, t):
            return s + t * (e - s)

        t = epoch / (total_epochs - 1)
        lw = cfg['linear_weights']
        return lerp(lw['w_cross']['start'], lw['w_cross']['end'], t), \
            lerp(lw['w_emg']['start'], lw['w_emg']['end'], t), \
            lerp(lw['w_imu']['start'], lw['w_imu']['end'], t)


# ==== Step 9: main ====
def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    data_cfg = config['data']
    train_cfg = config['train']
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")

    knee_pad = KneePAD(root_dir=data_cfg['knee_pad_root'])
    move_port = MovePort(root_dir=data_cfg['move_port_root'])
    train_loader, val_loader = build_phase2_loaders(config, [knee_pad, move_port])

    emg_encoder = load_pretrained_encoder(create_emg_encoder(config), "time_encoder_emg.pt", freeze_cnn=True)
    imu_encoder = load_pretrained_encoder(create_imu_encoder(config), "time_encoder_imu.pt", freeze_cnn=True)

    model = MultiModalSSL(emg_encoder, imu_encoder, config['model']).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(train_cfg['lr']))

    train_phase2(train_cfg, train_loader, model, optimizer)


if __name__ == "__main__":
    main()

