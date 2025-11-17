import torch
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import create_dataloaders
from ssl_model import create_emg_encoder, create_imu_encoder
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


def load_encoder(modality, config):
    encoder = create_emg_encoder(config) if modality == 'emg' else create_imu_encoder(config)
    encoder_path = Path(config['train']['output_dir']) / 'models' / f"time_encoder_{modality}.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder weights not found at {encoder_path}")
    state_dict = torch.load(encoder_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def extract_features_labels(dataloader, encoder, modality, label_enc=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    features, label_names = [], []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            x = batch.get('emg') if modality == 'emg' else batch.get('imu')
            if x is None:
                continue
            x = x.to(device)

            z = encoder(x)
            if z.ndim == 3:
                z = z.mean(dim=1)
            features.append(z.cpu().numpy())

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

    features = np.vstack(features)
    if label_enc is None:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(label_names)
    else:
        y = label_enc.transform(label_names)

    print(f"features: {features.shape}, labels: {len(label_names)}")

    return features, y, label_enc


def plot_confusion(y_true, y_pred, label_encoder, title, save_path):
    labels = np.arange(len(label_encoder.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def run_eval(X_train, y_train, X_val, y_val, label_encoder, output_dir, clf_name, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)

    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
    logger.info(f"{clf_name} Evaluation Report:\n{report}")

    # cm_path = Path(output_dir) / f"{clf_name}_confusion_matrix.png"
    # plot_confusion(y_val, y_pred, label_encoder, f"{clf_name} Confusion Matrix", cm_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate pretrained encoder on downstream classification")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--modality', choices=['emg', 'imu'], default='emg')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataloaders = create_dataloaders(config, args.modality)
    train_loader, val_loader = dataloaders['train'], dataloaders['val']

    encoder = load_encoder(args.modality, config)

    logger.info("Extracting features for train set...")
    X_train, y_train, label_enc = extract_features_labels(train_loader, encoder, args.modality)
    logger.info("Extracting features for val set...")
    X_val, y_val, _ = extract_features_labels(val_loader, encoder, args.modality, label_enc)

    # ==== 只保留默认 LogReg ====
    run_eval(X_train, y_train, X_val, y_val, label_enc, config['train']['output_dir'],
             "LogReg_Default", LogisticRegression(max_iter=500))
