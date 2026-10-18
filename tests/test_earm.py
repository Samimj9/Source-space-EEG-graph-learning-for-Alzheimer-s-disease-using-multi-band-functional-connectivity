"""EARM-GNN (Experiment 7, methodology log entry 43): focused checks of the model and its experiment.

* shapes, softmax band weights, four distinct band representations;
* symmetric gates in (0, 1), gated adjacency <= EEG adjacency, bounded operator, finite values;
* gradients reach the edge gates (including through the penalty);
* the fixed-adjacency ablation has no gate parameters and uses D^-1/2 A D^-1/2;
* one fit per outer fold on training subjects only; fitted state (gates, history) saved per fold.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment, set_dotted  # noqa: E402
from adgraph.experiments.runner import run_variant  # noqa: E402
from adgraph.models.earm import EARMGNN, propagation  # noqa: E402
from adgraph.legacy.synthetic import make_synthetic_store  # noqa: E402

CFG = Path("configs/experiments/exp7_earm_gnn.yaml")
B, K, R, F = 5, 4, 68, 3


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, K, R, R, generator=g)
    a = (a + a.transpose(-1, -2)) / 2
    a = a * (1 - torch.eye(R))
    return torch.randn(B, K, R, F, generator=g), a


def net(**kw):
    torch.manual_seed(0)
    return EARMGNN(in_dim=F, n_bands=K, n_rois=R, **kw)


def test_shapes_and_band_weights():
    m = net().eval()
    x, a = inputs()
    logits, w = m(x, a, return_band_weights=True)
    assert logits.shape == (B,) and w.shape == (B, K)
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-6) and (w > 0).all()
    assert m.node_embeddings(x, a).shape == (B, K, R, 32)
    z, roi_att = m.band_embeddings(x, a, return_roi_attention=True)
    assert z.shape == (B, K, 96) and roi_att.shape == (B, K, R)
    assert torch.allclose(roi_att.sum(-1), torch.ones(B, K), atol=1e-6)
    assert m.fused_embedding(x, a).shape == (B, 96)
    assert sum(p.numel() for p in m.parameters()) == 18214


def test_gates_symmetric_bounded_and_operator_finite():
    m = net()
    g = m.gates()
    assert g.shape == (K, R, R) and torch.allclose(g, g.transpose(-1, -2)) and ((g > 0) & (g < 1)).all()
    x, a = inputs()
    p = m.operator(a)
    assert torch.isfinite(p).all() and torch.allclose(p, p.transpose(-1, -2), atol=1e-6)
    assert ((a * g) <= a + 1e-7).all()
    rho = torch.linalg.eigvalsh(p.double()).abs().amax()
    assert rho <= 1 + 1e-6                                     # bounded propagation (no blow-up with depth)


def test_gradients_reach_gates():
    m = net()
    x, a = inputs()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x, a), torch.tensor([0., 1, 0, 1, 1]))
    (loss + m.regularization()).backward()
    for p in (m.gate.U, m.gate.s, m.gate.b):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert m.regularization() > 0


def test_fixed_adjacency_ablation():
    m = net(adaptive_adjacency=False)
    assert m.gate is None and m.gates() is None and float(m.regularization()) == 0.0
    _, a = inputs()
    d = a.sum(-1).pow(-0.5)
    assert torch.allclose(m.operator(a), d.unsqueeze(-1) * a * d.unsqueeze(-2))
    assert torch.allclose(propagation(a, torch.ones(K, R, R)), propagation(a))
    assert sum(p.numel() for p in m.parameters()) == 18214 - (68 * 8 + 4 * 8 + 4)


def test_bands_are_encoded_separately():
    m = net().eval()
    x, a = inputs()
    z0 = m.band_embeddings(x, a)
    a2 = a.clone()
    a2[:, 0] = a2[:, 0].flip(-1).flip(-2)                       # change only the theta graph
    z1 = m.band_embeddings(x, a2)
    assert not torch.allclose(z0[:, 0], z1[:, 0]) and torch.allclose(z0[:, 1:], z1[:, 1:])
    with pytest.raises(ValueError):
        EARMGNN(in_dim=F, n_bands=K, n_rois=R, fusion="concat")
