from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.use_edge_attr = True

        for _ in range(num_layers):
            try:
                layer = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            except TypeError:
                self.use_edge_attr = False
                layer = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            self.gat_layers.append(layer)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_proj(x)
        x = F.elu(x)

        for gat_layer in self.gat_layers:
            if self.use_edge_attr:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.regressor(x)
        return out.view(-1)
