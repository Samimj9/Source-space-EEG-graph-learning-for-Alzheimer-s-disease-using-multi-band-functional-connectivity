"""Experiment 8: Xu-inspired dual-adjacency source-space GCN (methodology log entry 52, docs/lcmv_xu.md §9).

An ADAPTATION of the two-layer design of Xu et al. (2025), Front. Neurosci. 19:1555657 (Eqs. 4-7, Fig. 3), to
our representation. It is neither a reproduction of Xu's MF-MGCN nor a replacement for Exp 0-7:

* graphs: 68 Desikan-Killiany LCMV source ROIs (not 19 electrodes), one graph per band (five Xu bands);
* node features: our stored per-window features (log / relative PSD, differential entropy, node strength);
* layer 1, functional: our weighted PLV or |ciPLV| of that band and window (not Xu's Pearson correlation of DE
  feature vectors; values are used as weights, never cast to integers or binarized);
* layer 2, anatomical: a FIXED R x R matrix from atlas labels (``anatomical_adjacency``): 1 if two ROIs belong
  to the same predefined lobar group (``lobe_hemisphere``, ``lobe_bilateral``) or, for ``surface_contiguity``
  (amendment A2, methodology log entry 60), if the two regions share an edge of the fsaverage cortical mesh
  (``source.atlas.contiguity``; never across hemispheres), else 0. This is atlas-derived group membership or
  cortical-surface neighbourhood, NOT measured (e.g. diffusion-MRI) white-matter structural connectivity. It depends
  only on the ROI names, never on diagnoses, data or predictions.

Per band b (band-specific weights, as drawn in Xu's Fig. 3; ``band_specific: false`` shares them, as in Xu's code):

    H1_b = ReLU( N(A_func_b) X_b W1_b + c1_b )         N(A) = D^-1/2 (A + I) D^-1/2   (Kipf & Welling)
    H2_b = ReLU( N(A_anat)  H1_b W2_b + c2_b )

then the node outputs of all bands are fused by flattening (R x L x J, Xu's Fig. 4), dropout, and fully connected
layers S -> F -> 1 logit (binary cross-entropy; equivalent to Xu's two-class softmax). Xu chose K, S and F on their
test set (paper Figs. 8-9), so the experiment config uses our pre-specified fixed-protocol widths instead
(K = S = F = 32); L = 2 is Xu's node-output size (Table 1, not tuned). No value is tuned on our data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Lobar grouping of the 34 Desikan-Killiany labels per hemisphere (FreeSurfer aparc names), following the
# conventional FreeSurfer lobar assignment; the cingulate and the insula are separate groups (they belong to no
# single lobe). Precentral and paracentral -> frontal; postcentral -> parietal; medial temporal regions -> temporal.
DK_LOBES = {
    "frontal": ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral", "frontalpole"),
    "parietal": ("superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"),
    "temporal": ("superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"),
    "occipital": ("lateraloccipital", "lingual", "cuneus", "pericalcarine"),
    "cingulate": ("rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"),
    "insula": ("insula",),
}
GROUPINGS = ("lobe_hemisphere", "lobe_bilateral")         # group-membership priors (roi_groups)
ANATOMICAL = GROUPINGS + ("surface_contiguity",)          # every anatomical prior of anatomical_adjacency


def roi_groups(roi_names: list[str], grouping: str = "lobe_hemisphere") -> list[str]:
    """Group label of every ROI (``<lobe>-<hemi>`` or ``<lobe>``) for Desikan-Killiany names or the regions of any
    coarser atlas of ``source.atlas`` (their lobe is defined there); raises for an unknown name."""
    if grouping not in GROUPINGS:
        raise ValueError(f"anatomical grouping must be one of {GROUPINGS}")
    from ..source import atlas

    lobe_of = {f"{r}-{h}": lobe for lobe, regions in DK_LOBES.items() for r in regions for h in ("lh", "rh")}
    for a in atlas.ATLASES:
        lobe_of.update(atlas.lobe_of(a))
    out = []
    for name in roi_names:
        hemi = name.rpartition("-")[2]
        if name not in lobe_of or hemi not in ("lh", "rh"):
            raise ValueError(f"ROI {name!r} is not a cortical region '<region>-<lh|rh>' of a known atlas")
        out.append(f"{lobe_of[name]}-{hemi}" if grouping == "lobe_hemisphere" else lobe_of[name])
    return out


def anatomical_adjacency(roi_names: list[str], grouping: str = "lobe_hemisphere") -> np.ndarray:
    """Fixed binary (R, R) matrix, diagonal 0 (self-loops are added by the normalization).
    ``lobe_hemisphere``: same lobe and same hemisphere (12 groups; anatomical proximity);
    ``lobe_bilateral``: same lobe in either hemisphere (6 groups; the analogue of Xu's electrode regions; for the
    12-lobe atlas it links each lobe to its contralateral homologue only);
    ``surface_contiguity``: the two regions share an edge of the fsaverage cortical mesh (``source.atlas.contiguity``,
    within a hemisphere; amendment A2). It is the only one of the three that is informative for every atlas: with
    ``lobe_hemisphere`` each region of the 12-lobe atlas is its own group and the matrix is empty."""
    if grouping == "surface_contiguity":
        from ..source import atlas

        a = atlas.contiguity(list(roi_names)).astype(np.float32)
        np.fill_diagonal(a, 0.0)
        if not a.any():
            raise ValueError("surface contiguity of these regions is empty: the anatomical layer would carry no "
                             "information")
        return a
    g = np.array(roi_groups(roi_names, grouping))
    a = (g[:, None] == g[None, :]).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    if not a.any():
        raise ValueError(f"anatomical grouping {grouping!r} puts every ROI in its own group (e.g. the 12-lobe atlas "
                         "with lobe_hemisphere): the anatomical layer would carry no information")
    return a


def normalize(adj: torch.Tensor) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 over the last two axes (weighted, non-negative A)."""
    a = adj + torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
    d = a.sum(-1).clamp_min(1e-12).rsqrt()
    return d[..., :, None] * a * d[..., None, :]


class BandGCN(nn.Module):
    """One graph-convolution layer per band (band-specific or shared weights): (B, K, R, Fin) -> (B, K, R, Fout)."""

    def __init__(self, n_bands: int, f_in: int, f_out: int, band_specific: bool):
        super().__init__()
        k = n_bands if band_specific else 1
        self.weight = nn.Parameter(torch.empty(k, f_in, f_out))
        self.bias = nn.Parameter(torch.zeros(k, f_out))
        for i in range(k):
            nn.init.xavier_uniform_(self.weight.data[i])

    def forward(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        w = self.weight.expand(h.shape[1], -1, -1) if self.weight.shape[0] == 1 else self.weight
        b = self.bias.expand(h.shape[1], -1) if self.bias.shape[0] == 1 else self.bias
        hw = torch.einsum("bkrf,kfo->bkro", h, w)
        return p @ hw + b[None, :, None, :]


class XuDualGCN(nn.Module):
    def __init__(self, in_dim: int, n_bands: int, n_rois: int, roi_names: list[str], hidden: int = 32,
                 node_out: int = 2, fc1: int = 32, fc2: int = 32, dropout: float = 0.3, band_specific: bool = True,
                 layer1: str = "functional", layer2: str = "anatomical", anatomical_grouping: str = "lobe_hemisphere"):
        super().__init__()
        for name, val in (("layer1", layer1), ("layer2", layer2)):
            if val not in ("functional", "anatomical"):
                raise ValueError(f"{name} must be 'functional' or 'anatomical'")
        if len(roi_names) != n_rois:
            raise ValueError(f"{len(roi_names)} ROI names for {n_rois} ROIs")
        self.n_bands, self.n_rois, self.layer1, self.layer2 = n_bands, n_rois, layer1, layer2
        self.register_buffer("a_anat", torch.from_numpy(anatomical_adjacency(list(roi_names), anatomical_grouping)))
        self.gcn1 = BandGCN(n_bands, in_dim, hidden, band_specific)
        self.gcn2 = BandGCN(n_bands, hidden, node_out, band_specific)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_bands * n_rois * node_out, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, 1)
        for lin in (self.fc1, self.fc2, self.out):          # as Xu's code: Kaiming-normal weights, zero bias
            nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(lin.bias)

    def _operator(self, kind: str, adj: torch.Tensor) -> torch.Tensor:
        if kind == "functional":
            return normalize(adj)                            # (B, K, R, R): the window's own band connectivity
        return normalize(self.a_anat)[None, None]           # (1, 1, R, R): fixed atlas matrix

    def fused_embedding(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, k, r, _ = x.shape
        if k != self.n_bands or r != self.n_rois:
            raise ValueError(f"model built for {self.n_bands} bands x {self.n_rois} ROIs, got {k} x {r}")
        h = F.relu(self.gcn1(x, self._operator(self.layer1, adj)))
        h = F.relu(self.gcn2(h, self._operator(self.layer2, adj)))
        z = self.drop(h.reshape(b, -1))
        return F.relu(self.fc2(F.relu(self.fc1(z))))

    def forward(self, x, adj, return_band_weights: bool = False):
        logits = self.out(self.fused_embedding(x, adj)).squeeze(-1)
        if return_band_weights:                              # no band attention: uniform (bands are concatenated)
            return logits, torch.full((x.shape[0], self.n_bands), 1.0 / self.n_bands, device=x.device)
        return logits
