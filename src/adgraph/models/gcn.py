"""Dense weighted GCN (Kipf & Welling, 2017).

H' = D^-1/2 (A + I) D^-1/2 H W, with A the weighted connectivity matrix.
With constant node features the first layer outputs row sums of the normalized
adjacency, so connectivity information reaches the readout.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoders import DenseEncoder


class DenseGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = adj + torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
        d = a.sum(-1).clamp(min=1e-12).pow(-0.5)
        a = d.unsqueeze(-1) * a * d.unsqueeze(-2)
        return a @ self.lin(x) + self.bias


def gcn_encoder(in_dim: int, hidden: int, n_layers: int, dropout: float, **_) -> DenseEncoder:
    dims = [in_dim] + [hidden] * n_layers
    return DenseEncoder([DenseGCNConv(dims[i], dims[i + 1]) for i in range(n_layers)], dropout)
