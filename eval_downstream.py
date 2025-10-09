import torch
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import create_dataloaders
from ssl_model import create_emg_encoder, create_imu_encoder
from logging_utils import setup_logger

logger = setup_logger()


def load_encoder(modality, config):
    encoder = create_emg_encoder(config) if modality == 'emg' else create_imu_encoder(config)
    encoder_path = Path(config['train']['output_dir']) / 'models' / f"time_encoder_{modality}.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder weights not found at {encoder_path}")
    state_dict = torch.load(encoder_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def extract_features_labels(dataloader, encoder, modality):
    features, labels = [], []
    label_names = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            if modality == 'emg':
                x = batch.get('emg', None)
            else:
                x = batch.get('imu', None)
            if x is None:
                continue
            z = encoder(x).mean(dim=1).cpu().numpy()
            features.append(z)

            meta = batch['metadata']
            if isinstance(meta, dict):
                label_names.extend(meta['activity_name'])
            elif isinstance(meta, list):
                label_names.extend([m['activity_name'] for m in meta])
            else:
                raise TypeError(f"Unexpected metadata type: {type(meta)}")

    features = np.vstack(features)
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(label_names)
    return features, y, label_enc


def plot_confusion(y_true, y_pred, label_encoder, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
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

    # 保存报告
    output_dir = Path(output_dir)
    with open(output_dir / f"{clf_name}_report.txt", "w") as f:
        f.write(report)

    cm_path = output_dir / f"{clf_name}_confusion_matrix.png"
    plot_confusion(y_val, y_pred, label_encoder, f"{clf_name} Confusion Matrix", cm_path)


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
    X_val, y_val, _ = extract_features_labels(val_loader, encoder, args.modality)

    run_eval(X_train, y_train, X_val, y_val, label_enc, config['train']['output_dir'],
             "LogReg_Default", LogisticRegression(max_iter=500))

    run_eval(X_train, y_train, X_val, y_val, label_enc, config['train']['output_dir'],
             "LogReg_Balanced", LogisticRegression(max_iter=500, class_weight="balanced"))

    run_eval(X_train, y_train, X_val, y_val, label_enc, config['train']['output_dir'],
             "MLP_256_relu", MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=300))
