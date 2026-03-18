from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

# ===== Setup paths =====
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drug_solubility_gnn.data_utils import (
    build_graph_dataset,
    create_data_splits,
    get_split_datasets,
    load_raw_dataset,
)
from drug_solubility_gnn.model import GATRegressor


# ===== Utils =====

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return mae, rmse, r2


# ===== Train / Eval =====

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            loss = criterion(out, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs

            all_preds.extend(out.cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())

    mae, rmse, r2 = compute_metrics(all_targets, all_preds)

    return total_loss / len(loader.dataset), mae, rmse, r2


# ===== Main =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Load data =====
    raw_df = load_raw_dataset(ROOT_DIR / "curated-solubility-dataset.csv")

    # Normalize target
    y_mean = raw_df["solubility"].mean()
    y_std = raw_df["solubility"].std()
    raw_df["solubility"] = (raw_df["solubility"] - y_mean) / y_std

    graph_data_list, _ = build_graph_dataset(raw_df)
    split_indices = create_data_splits(graph_data_list, seed=args.seed)
    train_dataset, val_dataset, test_dataset = get_split_datasets(graph_data_list, split_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    sample = graph_data_list[0]
    edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else 0

    # ===== Model =====
    model = GATRegressor(
        in_channels=sample.num_node_features,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ROOT_DIR / "models" / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

    # ===== Final Test =====
    test_loss, test_mae, test_rmse, test_r2 = evaluate(model, test_loader, criterion, device)

    print("\n===== FINAL TEST =====")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
