"""Shared pieces of the dense GNN encoders.

All encoders take node features ``x`` (B, R, F) and a dense, non-negative,
symmetric weighted adjacency ``adj`` (B, R, R) and return node embeddings
(B, R, H). Dense batched operations are used because every graph has the same
68 nodes and is fully connected with weighted edges.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseEncoder(nn.Module):
    """Stack of ``n_layers`` graph convolutions with ReLU and dropout between them."""

    def __init__(self, convs: list[nn.Module], dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = self.dropout(torch.relu(x))
        return torch.relu(x)


ENCODERS = ("gcn",)


def build_encoder(name: str, in_dim: int, hidden: int, n_layers: int, dropout: float, **kwargs) -> DenseEncoder:
    """``gcn``: the standard weighted-adjacency GCN, the only encoder of the final experiments (EdgeWeightedGAT and
    WeightedGraphSAGE of the earlier Tier 2 analyses are in ``legacy/``)."""
    from .gcn import gcn_encoder

    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; the final framework uses {ENCODERS}")
    return gcn_encoder(in_dim, hidden, n_layers, dropout, **kwargs)
