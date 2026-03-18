from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.elu(self.input_proj(x))

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x  # skip connection

            x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x)

            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = x + residual  # residual connection

        x = global_mean_pool(x, batch)
        out = self.regressor(x)

        return out.view(-1)