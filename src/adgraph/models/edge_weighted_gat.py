"""EdgeWeightedGAT: dense graph attention with edge weights in attention and messages.

This is a *modified* GAT, not the canonical unweighted GAT of Velickovic et al. (2018),
and must be reported as "edge-weighted GAT" (EdgeWeightedGAT) in the paper.

Attention logits for edge j -> i in head h:
    e_ij = LeakyReLU(a_src . W h_i + a_dst . W h_j + c_h * A_ij)
softmax over neighbours j (edges with A_ij > 0, plus the self-loop).

Messages: m_ij = W h_j + A_ij * u_h when ``edge_in_message`` (default), else W h_j.

Why the edge terms: on a fully connected graph plain GAT ignores the edge weights,
and with constant node features every message W h_j is identical, so the output
would be the same for every graph. The learnable edge terms (c_h in the logits,
u_h in the messages) make the layer depend on functional connectivity, which is
what Experiment 2 (minimal node features) tests.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import DenseEncoder


class EdgeWeightedGATConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True, dropout: float = 0.0,
                 edge_in_message: bool = True):
        super().__init__()
        if concat and out_dim % heads:
            raise ValueError("out_dim must be divisible by heads when concatenating")
        self.heads, self.concat = heads, concat
        self.head_dim = out_dim // heads if concat else out_dim
        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_edge = nn.Parameter(torch.ones(heads))
        self.edge_msg = nn.Parameter(torch.empty(heads, self.head_dim)) if edge_in_message else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.attn_dropout = nn.Dropout(dropout)
        for p in (self.a_src, self.a_dst) + ((self.edge_msg,) if edge_in_message else ()):
            nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_attention: bool = False):
        b, r, _ = x.shape
        h = self.lin(x).view(b, r, self.heads, self.head_dim)                      # (B, R, H, D)
        e_i = (h * self.a_src).sum(-1)                                             # (B, R, H)
        e_j = (h * self.a_dst).sum(-1)
        logits = e_i.unsqueeze(2) + e_j.unsqueeze(1) + adj.unsqueeze(-1) * self.a_edge   # (B, R, R, H)
        logits = F.leaky_relu(logits, 0.2)
        eye = torch.eye(r, dtype=torch.bool, device=x.device)
        mask = (adj > 0) | eye                                                      # (B, R, R)
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = self.attn_dropout(torch.softmax(logits, dim=2))                     # over neighbours j
        out = torch.einsum("bijh,bjhd->bihd", alpha, h)
        if self.edge_msg is not None:
            out = out + torch.einsum("bijh,bij->bih", alpha, adj).unsqueeze(-1) * self.edge_msg
        out = out.reshape(b, r, -1) if self.concat else out.mean(2)
        out = out + self.bias
        return (out, alpha) if return_attention else out


def edge_weighted_gat_encoder(in_dim: int, hidden: int, n_layers: int, dropout: float, heads: int = 4,
                edge_in_message: bool = True, **_) -> DenseEncoder:
    dims = [in_dim] + [hidden] * n_layers
    convs = [EdgeWeightedGATConv(dims[i], dims[i + 1], heads=heads, concat=True, dropout=dropout,
                       edge_in_message=edge_in_message) for i in range(n_layers)]
    return DenseEncoder(convs, dropout)
