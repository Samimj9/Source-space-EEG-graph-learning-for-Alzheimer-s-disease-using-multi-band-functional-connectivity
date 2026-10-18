"""Frozen historical graph builder, kept importable and tested so that earlier results stay reproducible
(CLAUDE.md rule 7). NOT used by the final framework (``adgraph.sourcegraphs`` builds the ds004504 LCMV / wMNE stores).

* ``pipeline``: the boundary-aware v4 (and v3) builder of the D1 / D2 graph runs (``graphs_v4_*``) and of the
  historical wMNE Dataset 3 runs (``graphsD3_v4_*``), configs ``configs/pipeline.yaml`` / ``configs/pipeline_d3.yaml``;
* ``amplitude``: the read-only amplitude diagnosis of that builder (``scripts/legacy/diagnose_amplitude.py``);
* ``synthetic``: the synthetic D1 / D2 store and EDF helpers of its tests.

The D1 / D2 experiment framework, its figures and the earlier per-suite pins are archived (not importable) under
``legacy/`` at the repository root; see ``docs/final/inventory.md``.
"""
