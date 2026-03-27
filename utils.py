# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:55:22 2026


"""



import os
import sys
import numpy as np
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch.utils.data import Dataset


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def normalize(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    mn = float(data.min())
    mx = float(data.max())
    denom = (mx - mn) if (mx > mn) else 1.0
    out = 255.0 * (data - mn) / denom
    out = np.rint(out)
    return out.astype(np.float32)


class DASDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.names_list = []

        if not os.path.isfile(self.names_file):
            raise FileNotFoundError(self.names_file + " does not exist!")

        with open(self.names_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.names_list.append(line)

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        parts = self.names_list[idx].split()
        rel_path = parts[0]
        label = int(parts[1])

        data_path = os.path.join(self.root_dir, rel_path.lstrip("/\\"))
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path + " does not exist!")

        mat = scio.loadmat(data_path)
        if "data" not in mat:
            raise KeyError(f"'data' not found in {data_path}. Keys={list(mat.keys())}")

        rawdata = mat["data"].astype(np.int32)
        data = normalize(rawdata)

        sample = {"data": data, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def pick_nhead(d_model: int, preferred: int = 4) -> int:
    for nh in [preferred, 8, 4, 2, 1]:
        if nh > 0 and d_model % nh == 0:
            return nh
    return 1


def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    avg_usage = gate_weights.mean(dim=0)
    target = torch.full_like(avg_usage, 1.0 / gate_weights.size(1))
    return ((avg_usage - target) ** 2).mean()


def train_one_step(model, train_x, train_y, optimizer, criterion, lambda_balance=0.01, use_balance=False):
    model.train()
    optimizer.zero_grad()

    _, logits, gate_weights = model(train_x)

    ce_loss = criterion(logits, train_y)
    lb_loss = load_balance_loss(gate_weights) if use_balance and gate_weights is not None else torch.tensor(0.0, device=train_x.device)
    loss = ce_loss + lambda_balance * lb_loss

    pred = torch.argmax(logits, dim=1)
    labels = train_y.detach().cpu().tolist()
    preds = pred.detach().cpu().tolist()

    loss.backward()
    optimizer.step()

    return labels, preds, loss.item(), ce_loss.item(), lb_loss.item()


def evaluate_model(model, loader, criterion, num_classes=6, lambda_balance=0.01, use_balance=False):
    model.eval()
    total_batch_num = 0
    val_loss = 0.0
    ce_loss_sum = 0.0
    lb_loss_sum = 0.0

    predictions = []
    labels = []
    feature_list = torch.tensor([])
    gate_list = []
    gate_label_list = []

    with torch.no_grad():
        for batch in loader:
            total_batch_num += 1

            batch_x = torch.tensor(batch["data"]).float() if isinstance(batch["data"], np.ndarray) else batch["data"].float()
            batch_y = torch.tensor(batch["label"]).long() if isinstance(batch["label"], (int, np.integer)) else batch["label"].long()

            if torch.cuda.is_available():
                batch_x = batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)

            feature, logits, gate_weights = model(batch_x)

            batch_label = batch_y.unsqueeze(1).float()
            feature_label = torch.cat((feature, batch_label), dim=1)

            if feature_list.numel() == 0:
                feature_list = feature_label.detach().cpu()
            else:
                feature_list = torch.cat((feature_list, feature_label.detach().cpu()), dim=0)

            ce_loss = criterion(logits, batch_y)

            if use_balance and gate_weights is not None:
                gate_list.append(gate_weights.detach().cpu())
                gate_label_list.append(batch_y.detach().cpu())
                lb_loss = load_balance_loss(gate_weights)
            else:
                lb_loss = torch.tensor(0.0, device=batch_x.device)

            loss = ce_loss + lambda_balance * lb_loss

            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.detach().cpu().tolist())
            labels.extend(batch_y.detach().cpu().tolist())

            val_loss += loss.item()
            ce_loss_sum += ce_loss.item()
            lb_loss_sum += lb_loss.item()

    accuracy = accuracy_score(labels, predictions)
    conf_mat = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    gate_mean = None
    gate_per_class = None

    if len(gate_list) > 0:
        gate_all = torch.cat(gate_list, dim=0)
        gate_labels_all = torch.cat(gate_label_list, dim=0)

        gate_mean = gate_all.mean(dim=0).numpy()
        num_experts = gate_all.shape[1]
        gate_per_class = np.zeros((num_classes, num_experts), dtype=np.float32)

        for c in range(num_classes):
            mask = (gate_labels_all == c)
            if mask.sum().item() > 0:
                gate_per_class[c] = gate_all[mask].mean(dim=0).numpy()

    return (
        accuracy,
        val_loss / max(total_batch_num, 1),
        ce_loss_sum / max(total_batch_num, 1),
        lb_loss_sum / max(total_batch_num, 1),
        feature_list,
        conf_mat,
        gate_mean,
        gate_per_class
    )


def save_training_history(out_dir, run_name, train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    history_dict = {
        "epoch": np.arange(1, len(train_loss_list) + 1, dtype=np.int32),
        "train_loss": np.array(train_loss_list, dtype=np.float32),
        "train_acc": np.array(train_acc_list, dtype=np.float32),
        "val_loss": np.array(test_loss_list, dtype=np.float32),
        "val_acc": np.array(test_acc_list, dtype=np.float32),
    }

    history_csv_path = os.path.join(out_dir, f"{run_name}_training_history.csv")
    history_npy_path = os.path.join(out_dir, f"{run_name}_training_history.npy")

    pd.DataFrame(history_dict).to_csv(history_csv_path, index=False)
    np.save(history_npy_path, history_dict)

    np.save(os.path.join(out_dir, f"{run_name}_train_loss_list.npy"), np.array(train_loss_list, dtype=np.float32))
    np.save(os.path.join(out_dir, f"{run_name}_train_acc_list.npy"), np.array(train_acc_list, dtype=np.float32))
    np.save(os.path.join(out_dir, f"{run_name}_val_loss_list.npy"), np.array(test_loss_list, dtype=np.float32))
    np.save(os.path.join(out_dir, f"{run_name}_val_acc_list.npy"), np.array(test_acc_list, dtype=np.float32))

    print(f"Training history saved to:\n- {history_csv_path}\n- {history_npy_path}")


def plot_training_curves(train_acc, train_loss, val_acc, val_loss, save_path):
    epochs_acc = range(len(train_acc))
    epochs_loss = range(len(train_loss))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(epochs_acc, train_acc, label="train", color="b")
    plt.plot(epochs_acc, val_acc, label="validation", color="r")
    plt.legend(loc="best")
    plt.title("Accuracy vs. epochs")
    plt.ylabel("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(epochs_loss, train_loss, "-", label="train", color="b")
    plt.plot(epochs_loss, val_loss, "-", label="validation", color="r")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def draw_gate_class_heatmap(gate_per_class, class_names, save_path):
    df = pd.DataFrame(
        gate_per_class,
        index=class_names,
        columns=["temporal_expert", "spatial_expert", "fusion_expert"]
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues", cbar=True)
    plt.title("Mean expert weights per class")
    plt.xlabel("Experts")
    plt.ylabel("Classes")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_confusion_matrix(confusion_mat, save_path):
    confusion_mat = np.array(confusion_mat, dtype=np.int64)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    df = pd.DataFrame(confusion_mat)

    class_names = ["background", "digging", "knocking", "watering", "shaking", "walking"]

    sns.heatmap(
        df,
        fmt="g",
        annot=True,
        robust=True,
        annot_kws={"size": 10},
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )

    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_ylabel("True label", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    total = confusion_mat.sum()
    acc = np.trace(confusion_mat) / (total + 1e-12)
    print("acc: %.4f" % acc)

    non_bg = total - confusion_mat[0].sum()
    nar = (confusion_mat[0].sum() - confusion_mat[0, 0]) / (non_bg + 1e-12)
    print("NAR: %.4f" % nar)

    events_total = total - confusion_mat[0].sum()
    fnr = confusion_mat[1:, 0].sum() / (events_total + 1e-12)
    print("FNR: %.4f" % fnr)

    column_sum = np.sum(confusion_mat, axis=0)
    row_sum = np.sum(confusion_mat, axis=1)

    print("column_sum:", column_sum)
    print("row_sum:", row_sum)

    for k in range(confusion_mat.shape[0]):
        precision = confusion_mat[k, k] / (column_sum[k] + 1e-12)
        recall = confusion_mat[k, k] / (row_sum[k] + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        print(f"precision_{k} ({k}): {precision:.3f}")
        print(f"recall_{k} ({k}): {recall:.3f}")
        print(f"f1_{k} ({k}): {f1:.3f}")