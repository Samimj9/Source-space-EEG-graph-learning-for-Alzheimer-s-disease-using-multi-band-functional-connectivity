"""Graph classifier over K band graphs (K = 1 for single-band and band-averaged graphs).

    x   (B, K, R, F)  node features per band graph
    adj (B, K, R, R)  weighted adjacency per band graph
    ->  logits (B,)

Per band k: GNN encoder (shared weights across bands, or one encoder per band)
-> node embeddings (B, R, H) -> readout -> graph embedding z_k (B, D).
The K band embeddings are fused:

    mean        z = mean_k z_k
    concat      z = [z_1, ..., z_K]                          (D * K)
    gate        g_k = sigmoid(w . z_k + c);  z = sum_k g_k z_k / sum_k g_k
                independent, sample-specific band gates
    attention   s_k = q . tanh(W z_k + b);   a = softmax_k(s);  z = sum_k a_k z_k
                bands compete (semantic attention, Wang et al., HAN 2019)

``gate`` and ``attention`` are the learned-fusion candidates; their band weights
are returned for interpretation. With K = 1 every fusion reduces to identity.

Band identity (methodology log entry 57): with a shared encoder and mean / attention fusion the model is invariant
to permuting the band graphs, because nothing tells the encoder which band it is processing (per-band z-scoring
also removes each band's typical level). ``band_identity: onehot`` appends a constant one-hot band indicator
(K extra node-feature columns, identical for every ROI of band k) to every band's node features before the encoder,
so the shared weights can act band-specifically. Default ``none`` keeps the original architecture unchanged.

Covariates (EEG + age models): ``cov`` (B, n_covariates), already standardized with
training statistics, is concatenated to the fused graph embedding immediately before
the final MLP classifier; it never enters the GNN encoders or the fusion.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoders import build_encoder

FUSIONS = ("mean", "concat", "gate", "attention")
READOUTS = ("mean", "max", "sum", "mean_max")
BAND_IDENTITY = ("none", "onehot")


def readout(h: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mean":
        return h.mean(1)
    if kind == "max":
        return h.amax(1)
    if kind == "sum":
        return h.sum(1)
    if kind == "mean_max":
        return torch.cat([h.mean(1), h.amax(1)], dim=-1)
    raise ValueError(f"unknown readout {kind!r}; choose from {READOUTS}")


class BandFusion(nn.Module):
    def __init__(self, kind: str, dim: int, n_bands: int, attn_dim: int = 32):
        super().__init__()
        if kind not in FUSIONS:
            raise ValueError(f"unknown fusion {kind!r}; choose from {FUSIONS}")
        self.kind, self.n_bands = kind, n_bands
        self.out_dim = dim * n_bands if kind == "concat" else dim
        if kind == "gate":
            self.gate = nn.Linear(dim, 1)
        elif kind == "attention":
            self.proj = nn.Linear(dim, attn_dim)
            self.query = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, K, D) -> fused (B, out_dim), band weights (B, K)."""
        b, k, _ = z.shape
        if self.kind == "mean" or k == 1:
            w = torch.full((b, k), 1.0 / k, device=z.device)
            return (z.reshape(b, -1) if self.kind == "concat" else z.mean(1)), w
        if self.kind == "concat":
            return z.reshape(b, -1), torch.full((b, k), 1.0 / k, device=z.device)
        if self.kind == "gate":
            g = torch.sigmoid(self.gate(z)).squeeze(-1)                       # (B, K)
            w = g / g.sum(1, keepdim=True).clamp(min=1e-12)
        else:
            w = torch.softmax(self.query(torch.tanh(self.proj(z))).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * z).sum(1), w


class MultiBandGNN(nn.Module):
    def __init__(self, encoder: str, in_dim: int, n_bands: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, shared_encoder: bool = True, fusion: str = "attention",
                 readout: str = "mean_max", mlp_hidden: int = 32, encoder_kwargs: dict | None = None,
                 n_covariates: int = 0, band_identity: str = "none"):
        super().__init__()
        kw = encoder_kwargs or {}
        if band_identity not in BAND_IDENTITY:
            raise ValueError(f"band_identity must be one of {BAND_IDENTITY}")
        self.n_bands, self.shared, self.readout_kind = n_bands, shared_encoder, readout
        self.band_identity = band_identity
        enc_in = in_dim + (n_bands if band_identity == "onehot" else 0)
        n_enc = 1 if shared_encoder else n_bands
        self.encoders = nn.ModuleList([build_encoder(encoder, enc_in, hidden, n_layers, dropout, **kw)
                                       for _ in range(n_enc)])
        emb = hidden * (2 if readout == "mean_max" else 1)
        self.fusion = BandFusion(fusion, emb, n_bands)
        self.n_covariates = n_covariates
        self.embed_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(self.fusion.out_dim + n_covariates, mlp_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))

    def embed(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Band graph embeddings (B, K, D)."""
        b, k, r, f = x.shape
        if k != self.n_bands:
            raise ValueError(f"model built for {self.n_bands} band graphs, got {k}")
        if self.band_identity == "onehot":                  # (B, K, R, F) -> (B, K, R, F + K)
            eye = torch.eye(k, device=x.device, dtype=x.dtype)[None, :, None, :].expand(b, k, r, k)
            x = torch.cat([x, eye], dim=-1)
            f = x.shape[-1]
        if self.shared:
            h = self.encoders[0](x.reshape(b * k, r, f), adj.reshape(b * k, r, r))
            return readout(h, self.readout_kind).view(b, k, -1)
        return torch.stack([readout(self.encoders[i](x[:, i], adj[:, i]), self.readout_kind) for i in range(k)], 1)

    def fused_embedding(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Fused graph embedding (B, D_fused): after band fusion, before covariates and the classifier."""
        return self.fusion(self.embed(x, adj))[0]

    def forward(self, x: torch.Tensor, adj: torch.Tensor, cov: torch.Tensor | None = None,
                return_band_weights: bool = False):
        fused, w = self.fusion(self.embed(x, adj))
        fused = self.embed_dropout(fused)
        if self.n_covariates:
            if cov is None or cov.shape != (fused.shape[0], self.n_covariates):
                raise ValueError(f"model expects covariates of shape (B, {self.n_covariates})")
            fused = torch.cat([fused, cov], dim=-1)
        elif cov is not None:
            raise ValueError("covariates given to a model built without them")
        logits = self.head(fused).squeeze(-1)
        return (logits, w) if return_band_weights else logits
