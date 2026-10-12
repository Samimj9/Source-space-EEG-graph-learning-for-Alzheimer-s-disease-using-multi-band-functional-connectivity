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


def build_encoder(name: str, in_dim: int, hidden: int, n_layers: int, dropout: float, **kwargs) -> DenseEncoder:
    from .gat import gat_encoder
    from .gcn import gcn_encoder
    from .graphsage import sage_encoder

    makers = {"gcn": gcn_encoder, "gat": gat_encoder, "graphsage": sage_encoder}
    if name not in makers:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(makers)}")
    return makers[name](in_dim, hidden, n_layers, dropout, **kwargs)
