"""WeightedGraphSAGE: dense GraphSAGE with edge-weighted neighbour aggregation.

This is a *modified* GraphSAGE (Hamilton et al., 2017), not the canonical unweighted
mean-aggregator GraphSAGE, and must be reported as "weighted GraphSAGE" in the paper.

h_i' = W_self h_i + W_neigh AGG_j(A_ij, h_j), optionally L2-normalized.

Aggregators:
    sum  (default)  sum_j A_ij h_j / (R - 1)          weighted sum, scaled by graph size
    mean            sum_j A_ij h_j / sum_j A_ij       weighted mean
    max             max_j A_ij h_j

The default is ``sum`` because a weighted *mean* of constant node features is the
same constant for every node and every graph: with the minimal (constant) node
features of Experiment 2, mean aggregation would discard the connectivity entirely.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import DenseEncoder

AGGREGATORS = ("sum", "mean", "max")


class WeightedSAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "sum", normalize: bool = False):
        super().__init__()
        if aggr not in AGGREGATORS:
            raise ValueError(f"aggr must be one of {AGGREGATORS}")
        self.aggr, self.normalize = aggr, normalize
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        r = adj.shape[-1]
        a = adj * (1 - torch.eye(r, device=adj.device, dtype=adj.dtype))   # neighbours only
        if self.aggr == "sum":
            agg = a @ x / (r - 1)
        elif self.aggr == "mean":
            agg = a @ x / a.sum(-1, keepdim=True).clamp(min=1e-12)
        else:
            agg = (a.unsqueeze(-1) * x.unsqueeze(1)).amax(dim=2)
        out = self.lin_self(x) + self.lin_neigh(agg)
        return F.normalize(out, dim=-1) if self.normalize else out


def weighted_graphsage_encoder(in_dim: int, hidden: int, n_layers: int, dropout: float, aggr: str = "sum",
                 normalize: bool = False, **_) -> DenseEncoder:
    dims = [in_dim] + [hidden] * n_layers
    return DenseEncoder([WeightedSAGEConv(dims[i], dims[i + 1], aggr, normalize) for i in range(n_layers)], dropout)
