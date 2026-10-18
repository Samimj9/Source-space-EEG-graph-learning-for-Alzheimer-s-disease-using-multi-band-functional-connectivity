"""EEG-Guided Adaptive Residual Multiband GNN (EARM-GNN), Experiment 7.

Exploratory extension developed AFTER the Tier 1 GCN results had been observed (methodology log
entry 43). It is not claimed to be novel or superior; it is evaluated with the same subject-level
protocol, splits and fixed-hyperparameter rules as the other GNNs, with two pre-specified ablations.

Inputs (as for every multi-band model):  x (B, K, R, F) node features, adj (B, K, R, R) EEG
connectivity (PLV or |ciPLV|; symmetric, zero diagonal, in [0, 1]); K = 4 bands, R = 68 ROIs,
F = 3 ([log PSD, relative PSD, node strength] of the band graph).

A. EEG-guided adaptive adjacency (shared low-rank symmetric edge gates)
       G_k = sigmoid(b_k + U diag(s_k) U^T)              U in R^{R x r}, s_k in R^r, b_k scalar
       A~_k = A_k (.) G_k                                 (elementwise; symmetric because G_k is)
       P_k  = D_k^{-1/2} A~_k D_k^{-1/2},   D_k = diag(sum_j A_k,ij)   (degree of the EEG prior A_k)
   Gates are window-independent parameters learned from training subjects only. Normalizing with
   the degree of the *ungated* EEG adjacency keeps the operator bounded (0 <= A~ <= A elementwise,
   so the spectral radius of P_k is <= 1) and lets a closed gate remove a neighbour's message
   instead of being undone by renormalization; the penalty lambda * mean_{k, i<j} G_k,ij therefore
   genuinely discourages connections (it is added to the training loss, `regularization()`).
   Self-loops are not part of P_k: the node's own state enters through a separate transformation (B).
   Ablation "fixed adjacency": G_k = 1 (no gate parameters, no penalty), P_k = D^{-1/2} A D^{-1/2}.

B. Residual graph encoder, shared across bands (L = 3 blocks), with a band embedding e_k:
       H^0_k = X_k W_in + b_in + e_k
       H^l_k = H^{l-1}_k + Dropout(GELU(LayerNorm(H^{l-1}_k W^l_self + b^l + P_k H^{l-1}_k W^l_nbr)))
       H_k   = [H^0_k, H^1_k, H^2_k, H^3_k] W_jk + b_jk      (jumping knowledge, concatenation)
   H^0 is a purely node-wise transformation of the regional features, so regional information
   reaches the readout without any neighbourhood averaging (residual path + JK).

F. Readout per band: attention pooling over ROIs plus mean and max pooling
       a_k = softmax_i(w2 . tanh(W1 h_k,i + c1)),   z_k = [sum_i a_k,i h_k,i, mean_i h_k,i, max_i h_k,i]

E. Band fusion (attention; ablation: mean), softmax-normalized band weights returned for interpretation
       s_k = q . tanh(W z_k + c),  beta = softmax_k(s),  z = sum_k beta_k z_k
   Classifier: Dropout -> Linear(3H, M) -> ReLU -> Dropout -> Linear(M, 1)  -> AD logit.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .multiband import BandFusion


class EdgeGate(nn.Module):
    """Symmetric, low-rank, band-specific edge gates G (K, R, R) in (0, 1)."""

    def __init__(self, n_rois: int, n_bands: int, rank: int = 8, init_bias: float = 2.0, init_std: float = 0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_rois, rank) * init_std)      # shared ROI embeddings
        self.s = nn.Parameter(torch.ones(n_bands, rank))                  # band-specific interaction weights
        self.b = nn.Parameter(torch.full((n_bands,), float(init_bias)))   # band-specific gate bias

    def logits(self) -> torch.Tensor:
        return self.b[:, None, None] + torch.einsum("ia,ka,ja->kij", self.U, self.s, self.U)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits())


def propagation(adj: torch.Tensor, gate: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """P = D^{-1/2} (A (.) G) D^{-1/2} with D the degree of the EEG adjacency A (no self-loops).

    adj (..., K, R, R); gate (K, R, R) or None (fixed adjacency)."""
    dinv = adj.sum(-1).clamp(min=eps).pow(-0.5)
    a = adj if gate is None else adj * gate
    return dinv.unsqueeze(-1) * a * dinv.unsqueeze(-2)


class ResidualGraphBlock(nn.Module):
    """H + Dropout(GELU(LayerNorm(H W_self + b + P H W_nbr)))."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.nbr_lin = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        z = self.self_lin(h) + p @ self.nbr_lin(h)
        return h + self.dropout(F.gelu(self.norm(z)))


class EARMGNN(nn.Module):
    def __init__(self, in_dim: int, n_bands: int, n_rois: int, hidden: int = 32, n_blocks: int = 3,
                 dropout: float = 0.3, adaptive_adjacency: bool = True, gate_rank: int = 8,
                 gate_init_bias: float = 2.0, gate_l1: float = 0.01, band_embedding: bool = True,
                 pool_hidden: int = 16, fusion: str = "attention", mlp_hidden: int = 32):
        super().__init__()
        if fusion not in ("attention", "mean"):
            raise ValueError("EARM-GNN fusion must be 'attention' (full model) or 'mean' (ablation)")
        self.n_bands, self.n_rois, self.gate_l1 = n_bands, n_rois, float(gate_l1)
        self.input = nn.Linear(in_dim, hidden)
        self.band_emb = nn.Parameter(torch.zeros(n_bands, hidden)) if band_embedding else None
        self.gate = EdgeGate(n_rois, n_bands, gate_rank, gate_init_bias) if adaptive_adjacency else None
        self.blocks = nn.ModuleList([ResidualGraphBlock(hidden, dropout) for _ in range(n_blocks)])
        self.jk = nn.Linear(hidden * (n_blocks + 1), hidden)
        self.pool_score = nn.Sequential(nn.Linear(hidden, pool_hidden), nn.Tanh(), nn.Linear(pool_hidden, 1))
        self.fusion = BandFusion(fusion, 3 * hidden, n_bands)
        self.embed_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3 * hidden, mlp_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))

    # -- components
    def gates(self) -> torch.Tensor | None:
        """Current edge gates (K, R, R), or None for the fixed-adjacency ablation."""
        return None if self.gate is None else self.gate()

    def operator(self, adj: torch.Tensor) -> torch.Tensor:
        return propagation(adj, self.gates())

    def regularization(self) -> torch.Tensor:
        """lambda * mean gate value over the distinct ROI pairs of every band (added to the loss)."""
        if self.gate is None or self.gate_l1 == 0:
            return torch.zeros((), device=self.input.weight.device)
        iu = torch.triu_indices(self.n_rois, self.n_rois, 1, device=self.input.weight.device)
        return self.gate_l1 * self.gate()[:, iu[0], iu[1]].mean()

    def node_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """(B, K, R, F), (B, K, R, R) -> (B, K, R, H) after the residual blocks and jumping knowledge."""
        b, k, r, _ = x.shape
        if k != self.n_bands or r != self.n_rois:
            raise ValueError(f"model built for {self.n_bands} bands x {self.n_rois} ROIs, got {k} x {r}")
        p = self.operator(adj)
        h = self.input(x)
        if self.band_emb is not None:
            h = h + self.band_emb[None, :, None, :]
        layers = [h]
        for blk in self.blocks:
            h = blk(h, p)
            layers.append(h)
        return self.jk(torch.cat(layers, dim=-1))

    def band_embeddings(self, x: torch.Tensor, adj: torch.Tensor, return_roi_attention: bool = False):
        """Band graph embeddings z (B, K, 3H) [and ROI attention a (B, K, R)]."""
        h = self.node_embeddings(x, adj)
        a = torch.softmax(self.pool_score(h).squeeze(-1), dim=-1)
        z = torch.cat([(a.unsqueeze(-1) * h).sum(-2), h.mean(-2), h.amax(-2)], dim=-1)
        return (z, a) if return_roi_attention else z

    # -- interface shared with MultiBandGNN (trainer.GNNModel)
    def fused_embedding(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.band_embeddings(x, adj))[0]

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_band_weights: bool = False):
        fused, w = self.fusion(self.band_embeddings(x, adj))
        logits = self.head(self.embed_dropout(fused)).squeeze(-1)
        return (logits, w) if return_band_weights else logits


def parameter_table(net: nn.Module) -> list[tuple[str, tuple, int]]:
    """(name, shape, count) of every trainable parameter."""
    return [(n, tuple(p.shape), p.numel()) for n, p in net.named_parameters() if p.requires_grad]
