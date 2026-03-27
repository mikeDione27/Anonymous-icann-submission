# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:00:45 2026

"""

# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils import (
    Logger,
    DASDataset,
    train_one_step,
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix,
    draw_gate_class_heatmap,
    save_training_history
)

from models import (
    CNNOnly,
    CNN1Transformer,
    DASTMoE
)


def build_model(args):
    if args.model_name == "CNNOnly":
        model = CNNOnly(
            n_sensors=args.n_sensors,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
        use_balance = False

    elif args.model_name == "CNN1Transformer":
        model = CNN1Transformer(
            n_sensors=args.n_sensors,
            cnn_emb=args.cnn_emb,
            hidden_dim=args.hidden_dim,
            temporal_dim=args.temporal_dim,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
            num_classes=args.num_classes,
            nhead=args.nhead,
            dropout=args.dropout,
            pooling=args.pooling
        )
        use_balance = False

    elif args.model_name == "DASTMoE":
        model = DASTMoE(
            n_sensors=args.n_sensors,
            cnn_emb=args.cnn_emb,
            hidden_dim=args.hidden_dim,
            temporal_dim=args.temporal_dim,
            num_classes=args.num_classes,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
            nhead=args.nhead,
            num_layers_expert=args.num_layers_expert,
            num_layers_shared=args.num_layers_shared,
            dropout=args.dropout,
            pooling=args.pooling
        )
        use_balance = True
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    return model, use_balance


def main(args):
    run_name = f"{args.model_name}_{args.dataset_name}"

    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.out_dir, f"{run_name}_result.log"), sys.stdout)

    train_dataset = DASDataset(args.root, args.txtpath, transform=None)
    test_dataset = DASDataset(args.root2, args.txtpath2, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model, use_balance = build_model(args)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    batches_per_epoch = max(int(len(train_dataset) / args.batch_size), 1)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    train_time = 0.0

    final_feature_list = None
    final_confusion_matrix = None
    final_gate_mean = None
    final_gate_per_class = None

    for epoch in range(args.epochs):
        tic = time.time()
        train_predictions = []
        train_labels = []
        running_loss = 0.0

        for batch in train_loader:
            batch_x = batch["data"].float()
            batch_y = batch["label"].long()

            if torch.cuda.is_available():
                batch_x = batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)

            labels, preds, loss_value, ce_loss, lb_loss = train_one_step(
                model=model,
                train_x=batch_x,
                train_y=batch_y,
                optimizer=optimizer,
                criterion=criterion,
                lambda_balance=args.lambda_balance,
                use_balance=use_balance
            )

            running_loss += loss_value
            train_labels.extend(labels)
            train_predictions.extend(preds)

        epoch_train_time = time.time() - tic
        train_time += epoch_train_time

        train_acc = accuracy_score(train_labels, train_predictions)
        train_loss = running_loss / batches_per_epoch

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        (
            val_acc,
            val_loss,
            ce_score,
            lb_score,
            feature_list,
            confusion_mat,
            gate_mean,
            gate_per_class
        ) = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            num_classes=args.num_classes,
            lambda_balance=args.lambda_balance,
            use_balance=use_balance
        )

        print(
            "Epoch %d Train_acc %.3f Train_loss %.3f | Val_acc %.3f Val_loss %.3f | CE %.3f LB %.6f"
            % (epoch, train_acc, train_loss, val_acc, val_loss, ce_score, lb_score)
        )

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        final_feature_list = feature_list
        final_confusion_matrix = confusion_mat
        final_gate_mean = gate_mean
        final_gate_per_class = gate_per_class

    model_path = os.path.join(args.out_dir, f"{run_name}_model.pth")
    feature_csv_path = os.path.join(args.out_dir, f"{run_name}_feature_data.csv")
    cm_path = os.path.join(args.out_dir, f"{run_name}_confusion_matrix.jpg")
    curves_path = os.path.join(args.out_dir, f"{run_name}_training_curves.jpg")

    torch.save(model, model_path)
    np.savetxt(feature_csv_path, final_feature_list.numpy(), delimiter=",")

    if use_balance and final_gate_mean is not None:
        gate_csv_path = os.path.join(args.out_dir, f"{run_name}_gate_mean_usage.csv")
        gate_df = pd.DataFrame(
            final_gate_mean.reshape(1, -1),
            columns=["temporal_expert", "spatial_expert", "fusion_expert"]
        )
        gate_df.to_csv(gate_csv_path, index=False)

        print("Final gate mean usage [temporal, spatial, fusion]:", final_gate_mean)

        if args.num_classes == 6:
            class_names = ["background", "digging", "knocking", "watering", "shaking", "walking"]
        else:
            class_names = [f"class_{i}" for i in range(args.num_classes)]

        if final_gate_per_class is not None:
            gate_class_csv_path = os.path.join(args.out_dir, f"{run_name}_gate_per_class.csv")
            gate_class_df = pd.DataFrame(
                final_gate_per_class,
                index=class_names,
                columns=["temporal_expert", "spatial_expert", "fusion_expert"]
            )
            gate_class_df.to_csv(gate_class_csv_path, index=True)

            print("\nMean expert weights per class:")
            print(gate_class_df)

            gate_heatmap_path = os.path.join(args.out_dir, f"{run_name}_gate_per_class_heatmap.jpg")
            draw_gate_class_heatmap(
                final_gate_per_class,
                class_names=class_names,
                save_path=gate_heatmap_path
            )

    plot_confusion_matrix(final_confusion_matrix, save_path=cm_path)
    plot_training_curves(
        train_acc=train_acc_list,
        train_loss=train_loss_list,
        val_acc=val_acc_list,
        val_loss=val_loss_list,
        save_path=curves_path
    )

    save_training_history(
        out_dir=args.out_dir,
        run_name=run_name,
        train_loss_list=train_loss_list,
        train_acc_list=train_acc_list,
        test_loss_list=val_loss_list,
        test_acc_list=val_acc_list
    )

    print("Total training time: %.3f seconds" % train_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAS multi-model training")

    rootpath = "./data"
    parser.add_argument("--root", type=str, default=rootpath + "/train", help="Path to training data")
    parser.add_argument("--root2", type=str, default=rootpath + "/test", help="Path to test data")
    parser.add_argument("--txtpath", type=str, default=rootpath + "/train/label.txt", help="Training label file")
    parser.add_argument("--txtpath2", type=str, default=rootpath + "/test/label.txt", help="Test label file")

    parser.add_argument("--out_dir", type=str, default="./outputs", help="Folder for outputs")
    parser.add_argument("--dataset_name", type=str, default="LaboDAS", help="Dataset name used in filenames")

    parser.add_argument(
        "--model_name",
        type=str,
        default="DASTMoE",
        choices=["CNNOnly", "CNN1Transformer", "DASTMoE"],
        help="Model to train"
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    parser.add_argument("--n_sensors", type=int, default=12, help="Number of DAS channels")
    parser.add_argument("--temporal_dim", type=int, default=10000, help="Temporal length")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes")

    parser.add_argument("--cnn_emb", type=int, default=256, help="Kept for compatibility")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Preferred transformer heads")
    parser.add_argument("--num_layers_expert", type=int, default=2, help="Number of expert layers")
    parser.add_argument("--num_layers_shared", type=int, default=2, help="Number of shared fusion layers")
    parser.add_argument("--stride", type=int, default=100, help="Unused; kept for compatibility")
    parser.add_argument("--kernel_size", type=int, default=15, help="Unused; kept for compatibility")
    parser.add_argument("--padding", type=int, default=7, help="Unused; kept for compatibility")
    parser.add_argument("--dilation", type=int, default=1, help="Unused; kept for compatibility")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--pooling", type=str, default="max", choices=["mean", "max"], help="Pooling mode")

    parser.add_argument("--lambda_balance", type=float, default=0.01, help="Load balancing loss weight")

    args = parser.parse_args()
    main(args)