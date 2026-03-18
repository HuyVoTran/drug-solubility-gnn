from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
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
from drug_solubility_gnn.metrics import compute_regression_metrics  # noqa: E402
from drug_solubility_gnn.model import GATRegressor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained GAT model on test set")
    parser.add_argument("--data-path", type=str, default=str(ROOT_DIR / "curated-solubility-dataset.csv"))
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "models" / "best_gat_model.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch)
            targets = batch.y.view(-1)

            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)
    return y_true, y_pred


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

    plt.xlabel("True LogS")
    plt.ylabel("Predicted LogS")
    plt.title("True vs Predicted LogS")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0.0, linestyle="--", linewidth=2)
    plt.xlabel("Predicted LogS")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(ROOT_DIR / "results", exist_ok=True)
    os.makedirs(ROOT_DIR / "plots", exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    feature_info = checkpoint["feature_info"]

    raw_df = load_raw_dataset(args.data_path)
    graph_data_list, _ = build_graph_dataset(raw_df)

    split_indices = checkpoint.get("split_indices")
    if split_indices is None:
        split_indices = create_data_splits(graph_data_list, seed=args.seed)

    _, _, test_dataset = get_split_datasets(graph_data_list, split_indices)
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GATRegressor(
        in_channels=int(feature_info["in_channels"]),
        edge_dim=int(feature_info["edge_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = collect_predictions(model, test_loader, device=device)
    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)

    print("Test metrics:")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")

    with open(ROOT_DIR / "results" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_prediction_plot(y_true, y_pred, ROOT_DIR / "plots" / "prediction_vs_true.png")
    save_residual_plot(y_true, y_pred, ROOT_DIR / "plots" / "residual_plot.png")

    print(f"Saved metrics: {ROOT_DIR / 'results' / 'test_metrics.json'}")
    print(f"Saved plot: {ROOT_DIR / 'plots' / 'prediction_vs_true.png'}")
    print(f"Saved plot: {ROOT_DIR / 'plots' / 'residual_plot.png'}")


if __name__ == "__main__":
    main()
