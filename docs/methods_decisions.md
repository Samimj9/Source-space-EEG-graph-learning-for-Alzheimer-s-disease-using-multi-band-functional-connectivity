# Methods decisions log

The single source of truth for methodological choices, especially
deviations from the reference papers ("Paper 1" = source-space pipeline
reference; "Paper 2" = the publication of Dataset 2; full citations TBD).
Add an entry whenever a parameter changes.

Seeded from `processed/run_manifest_20260912_162006.json`.

| # | Stage | Decision | Reference choice | Reason | Status |
|---|---|---|---|---|---|
| 1 | Preprocessing | Resample to 250 Hz; band-pass 0.1–45 Hz | | | adopted |
| 2 | Preprocessing | Average reference | not specified in Paper 1 | | adopted |
| 3 | Preprocessing | `standard_1020` montage, matched by channel name | landmark-based coregistration | no digitization available | adopted (limitation) |
| 4 | Preprocessing | D1 EDF ×1e-6 rescale | n/a | headers mislabel µV as V | adopted |
| 5 | Preprocessing | Drop M1/M2 in D1 only | | | **open**: asymmetric with D2 |
| 6 | Windowing | 40 s windows, 20 s step, 10 s edge trim; keep subject if ≥5 windows | | | adopted |
| 7 | Rejection | Per-subject adaptive threshold: median + 5·MAD of peak-to-peak; reject window if >15 % of channels exceed it | fixed ±80 µV | the fixed threshold did not transfer to this data | adopted |
| 8 | Source | fsaverage, ico-5 (20 484 vertices), 3-layer BEM 5120 | ICBM152 / Brainstorm | MNE-Python toolchain | adopted |
| 9 | Source | MNE inverse, loose 0.2, depth 0.8, λ² = 1/9 | | | adopted |
| 10 | Source | Desikan-Killiany (aparc), 68 ROIs, `mean_flip` | | | adopted |
| 11 | Connectivity | PLV, theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30 Hz | | | adopted |
| 12 | Data | D1 duplicate/mislabelled files | n/a | see `data_inventory.md` §1 | **open: blocking** |
| 13 | Evaluation | Subject-grouped CV, subject-level metrics primary | | windows are not independent | adopted |
