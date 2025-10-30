import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import yaml
from sklearn.preprocessing import LabelEncoder

from demo2 import build_phase2_loaders  # Phase2 数据加载器
import os
from datasets import KneePAD, MovePort

from logging_utils import setup_logger

logger = setup_logger()

# ==== 大类映射字典 ====
CLASS_MAP = {
    # Squat 系列
    "Squat": "Squat", "Squat_FL": "Squat", "Squat_WT": "Squat",
    # Extension 系列
    "Extension": "Extension", "Extension_NF": "Extension", "Extension_LL": "Extension",
    # Gait 系列
    "Gait": "Gait", "Gait_NF": "Gait", "Gait_HA": "Gait",
    # 其它不合并，维持原名
    "back": "back", "forward": "forward", "halfsquat": "halfsquat",
    "still": "still", "treadmill_dragging": "treadmill_dragging", "treadmill_leghigh": "treadmill_leghigh"
}


def extract_features(model, dataloader, device, mode ,label_enc=None):
    model.eval()
    feats = []
    label_names = []

    with torch.no_grad():
        for batch in dataloader:
            emg = batch['emg'].to(device)
            imu = batch['imu'].to(device)

            out = model(emg, imu)
            if mode == 'fusion':
                emb = out['fused_emb']
            elif mode == 'emg_raw':
                emb = out['emg_cls_raw']
            elif mode == 'imu_raw':
                emb = out['imu_cls_raw']
            else:
                raise ValueError(f"Unknown mode: {mode}")

            feats.append(emb.cpu().numpy())

            meta = batch['metadata']
            if isinstance(meta, dict) and isinstance(meta.get('activity_name'), (list, tuple)):
                names = [str(name) for name in meta['activity_name']]
            elif isinstance(meta, list):
                names = [str(m['activity_name']) for m in meta]
            else:
                names = [str(meta.get('activity_name', 'unknown'))]

            # 应用大类映射
            mapped_names = [CLASS_MAP.get(name, name) for name in names]
            label_names.extend(mapped_names)

    if label_enc is None:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(label_names)
    else:
        y = label_enc.transform(label_names)

    return np.vstack(feats), y, label_enc


def train_and_eval(X_train, y_train, X_val, y_val, label_encoder, title):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"\n{'=' * 30} {title} {'=' * 30}")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    report_dict = classification_report(y_val, y_pred, target_names=label_encoder.classes_, output_dict=True)
    return report_dict['accuracy'], report_dict['macro avg']['f1-score']


def main():
    # === 读取配置 ===
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config['data']
    train_cfg = config['train']
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")

    # === 数据加载（与训练一致） ===
    knee_pad = KneePAD(root_dir=data_cfg['knee_pad_root'])
    move_port = MovePort(root_dir=data_cfg['move_port_root'])
    train_loader, val_loader = build_phase2_loaders(config, [knee_pad, move_port])

    # === 加载模型（与训练一致） ===
    from demo1 import MultiModalSSL
    from ssl_model import create_emg_encoder, create_imu_encoder

    emg_encoder = create_emg_encoder(config)
    imu_encoder = create_imu_encoder(config)
    model = MultiModalSSL(emg_encoder, imu_encoder, config['model']).to(device)

    # 加载 Phase2 权重
    ckpt_path = 'phase2_final_model.pt'
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Loading Phase2 model checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    else:
        print("[WARN] No checkpoint specified or found, using random init.")

    # === 下游验证 ===
    results_table = []
    for mode in ['fusion', 'emg_raw', 'imu_raw']:
        print(f"[INFO] Extracting mode: {mode}")

        logger.info("Extracting features for train set...")
        X_train, y_train, label_enc = extract_features(model, train_loader, device, mode)

        logger.info("Extracting features for val set...")
        X_val, y_val, _ = extract_features(model, val_loader, device, mode)

        acc, f1 = train_and_eval(X_train, y_train, X_val, y_val, label_enc, mode)
        results_table.append((mode, acc, f1))

    # === 输出横向对比 ===
    print("\n=== Downstream Evaluation Summary ===")
    print("{:<10} | {:<8} | {:<8}".format("Mode", "Acc", "F1-macro"))
    print("-" * 30)
    for mode, acc, f1 in results_table:
        print("{:<10} | {:<8.3f} | {:<8.3f}".format(mode, acc, f1))


if __name__ == "__main__":
    main()
