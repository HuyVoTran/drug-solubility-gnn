from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool


class ImprovedGATRegressor(nn.Module):
    """
    Improved GAT-based regressor for drug solubility prediction.

    Key improvements over basic GAT:
    - Uses GATv2Conv for better attention mechanism
    - Increased model capacity (hidden_dim 128-256, num_layers 4-6, heads 4-8)
    - Residual connections for better gradient flow
    - LayerNorm after each GAT layer for stabilization
    - Configurable global pooling (mean or add)
    - Deeper MLP head with 3 layers
    - Multi-hop attention via stacked layers with residuals (inspired by MoGAT)
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.3,
        global_pool: str = "mean",  # "mean" or "add"
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if global_pool not in ["mean", "add"]:
            raise ValueError("global_pool must be 'mean' or 'add'")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pool = global_pool

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GAT layers with GATv2Conv
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()  # For residual connections

        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            out_dim = hidden_dim

            # GATv2Conv layer
            layer = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                share_weights=False,  # Allow different weights per layer
            )
            self.gat_layers.append(layer)

            # LayerNorm for stabilization
            self.norms.append(nn.LayerNorm(out_dim))

            # Residual projection if needed (for dimension matching)
            if i > 0:
                self.residual_projs.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.residual_projs.append(nn.Identity())

        # Global pooling selection
        self.pool_fn = global_mean_pool if global_pool == "mean" else global_add_pool

        # Final MLP head: 3 layers with ReLU and dropout
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)

        # Stacked GAT layers with residuals and norms
        for i, (gat_layer, norm, res_proj) in enumerate(zip(self.gat_layers, self.norms, self.residual_projs)):
            # Save input for residual
            x_res = x

            # GAT layer
            x = gat_layer(x, edge_index, edge_attr=edge_attr)

            # LayerNorm
            x = norm(x)

            # ELU activation
            x = F.elu(x)

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:  # Skip for first layer
                x = x + res_proj(x_res)

        # Global pooling
        x = self.pool_fn(x, batch)

        # Final regression
        out = self.regressor(x)
        return out.view(-1)
