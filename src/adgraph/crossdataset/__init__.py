"""Cross-dataset generalization D3 <-> D2 on a matched 19-electrode representation (docs/cross_dataset/design.md).

* ``channels``  the verified 19-electrode mapping (D2 EDF index -> D3 label) and the shared head geometry;
* ``d2store``   the matched-channel D2 source-graph stores (LCMV / wMNE, modes B / A, lobes12) in the isolated
                namespace ``<processed>/CROSS_D3_D2_19CH_v1/``;
* ``namespace`` paths, code pin, recorded authorizations and the write guard of that namespace;
* ``compat``    the D3 / D2 compatibility report and the harmonization manifest;
* ``plan``      the pre-specified campaign (directions x contexts x experiments x variants x seeds), its audit of
                earlier decisions and the frozen record written before any target label is read;
* ``transfer``  source-only fits and label-blind target predictions (checkpointed), source-only CV;
* ``evaluate``  participant-level evaluation after the freeze: metrics, participant bootstrap, transfer degradation,
                paired exploratory comparisons (Holm), age / sex analyses.

Nothing here writes outside the namespace; the release (FINAL_v1), amendment A2, the D3 stores, their lock, pins,
decisions and checkpoints are only read.
"""
