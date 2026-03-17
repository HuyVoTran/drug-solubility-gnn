from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drug_solubility_gnn.data_utils import (  # noqa: E402
    build_graph_dataset,
    create_data_splits,
    get_split_datasets,
    load_raw_dataset,
)
from drug_solubility_gnn.metrics import compute_accuracy  # noqa: E402
from drug_solubility_gnn.model import GATRegressor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT model for aqueous solubility prediction")
    parser.add_argument("--data-path", type=str, default=str(ROOT_DIR / "curated-solubility-dataset.csv"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs-before-stop", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accuracy-threshold", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        outputs = model(batch)
        targets = batch.y.view(-1)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_graphs = batch.num_graphs
        total_loss += float(loss.item()) * batch_graphs
        total_graphs += batch_graphs

    return total_loss / max(total_graphs, 1)


def evaluate_epoch(model, loader, criterion, device="cpu", accuracy_threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_graphs = 0
    total_correct = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y.view(-1)
            loss = criterion(outputs, targets)

            batch_graphs = batch.num_graphs
            total_loss += float(loss.item()) * batch_graphs
            total_graphs += batch_graphs
            batch_accuracy = compute_accuracy(
                y_true=targets.detach().cpu().numpy(),
                y_pred=outputs.detach().cpu().numpy(),
                threshold=accuracy_threshold,
            )
            total_correct += batch_accuracy * batch_graphs

    epoch_loss = total_loss / max(total_graphs, 1)
    epoch_accuracy = total_correct / max(total_graphs, 1)
    return epoch_loss, epoch_accuracy


def save_loss_plot(train_losses, val_losses, output_path: Path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss (MSE)", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss (MSE)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_accuracy_plot(train_accuracies, val_accuracies, output_path: Path):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(ROOT_DIR / "models", exist_ok=True)
    os.makedirs(ROOT_DIR / "results", exist_ok=True)
    os.makedirs(ROOT_DIR / "plots", exist_ok=True)

    raw_df = load_raw_dataset(args.data_path)
    graph_data_list, _ = build_graph_dataset(raw_df)
    split_indices = create_data_splits(graph_data_list, seed=args.seed)
    train_dataset, val_dataset, test_dataset = get_split_datasets(graph_data_list, split_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = graph_data_list[0]
    edge_dim = int(sample.edge_attr.size(-1)) if sample.edge_attr.numel() > 0 else 0

    model = GATRegressor(
        in_channels=sample.num_node_features,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    optimization_losses = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        optimization_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
        )
        train_loss, train_accuracy = evaluate_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            accuracy_threshold=args.accuracy_threshold,
        )
        val_loss, val_accuracy = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            accuracy_threshold=args.accuracy_threshold,
        )

        scheduler.step(val_loss)

        optimization_losses.append(optimization_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} "
            f"| Train Acc: {train_accuracy:.6f} | Val Acc: {val_accuracy:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "heads": args.heads,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "min_epochs_before_stop": args.min_epochs_before_stop,
                    "seed": args.seed,
                    "accuracy_threshold": args.accuracy_threshold,
                },
                "feature_info": {
                    "in_channels": int(sample.num_node_features),
                    "edge_dim": edge_dim,
                },
                "split_indices": split_indices,
                "best_epoch": best_epoch,
                "best_val_mse": best_val_loss,
            }
            torch.save(checkpoint, ROOT_DIR / "models" / "best_gat_model.pt")
        else:
            patience_counter += 1

        if epoch >= args.min_epochs_before_stop and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    save_loss_plot(train_losses, val_losses, ROOT_DIR / "plots" / "training_loss.png")
    save_accuracy_plot(train_accuracies, val_accuracies, ROOT_DIR / "plots" / "training_accuracy.png")

    train_summary = {
        "num_samples": {
            "total": len(graph_data_list),
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "optimization_loss_history": optimization_losses,
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "train_accuracy_history": train_accuracies,
        "val_accuracy_history": val_accuracies,
    }

    with open(ROOT_DIR / "models" / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    print("Training completed.")
    print(f"Best epoch: {best_epoch}, Best val MSE: {best_val_loss:.6f}")
    print(f"Saved model: {ROOT_DIR / 'models' / 'best_gat_model.pt'}")
    print(f"Saved plot: {ROOT_DIR / 'plots' / 'training_loss.png'}")
    print(f"Saved plot: {ROOT_DIR / 'plots' / 'training_accuracy.png'}")


if __name__ == "__main__":
    main()
