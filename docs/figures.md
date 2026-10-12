# Figures

This document lists every planned publication figure. The inventory itself is in
`configs/figures/paper.yaml`. Code is in `src/adgraph/visualization/`, and the driver
is `scripts/make_figures.py` (or `notebooks/03_figures.ipynb`).

**No figure has been generated from real data.**
- The figures are built only from saved outputs: graph runs, experiment output
  directories, an optional statistics CSV, and optional embedding files.
- No plotted number is typed into code or config.
- The test suite and `--synthetic-demo` exercise the code on synthetic outputs.
  Those figures carry the watermark "SYNTHETIC DATA – NOT RESULTS".

## General rules (apply to every figure)

| Rule | Implementation |
|---|---|
| Subject is the statistical unit | Window scores are aggregated to one score per subject before any metric, ROC, confusion matrix or distribution. PSD/connectivity are averaged within subject before group summaries. Windows are never plotted as observations. |
| Uncertainty | Performance points are CV repeats. Each repeat's metric uses all subjects' out-of-fold predictions. Bars show either SD across repeats, or a 95 % subject-bootstrap CI centred on the metric of repeat-averaged subject scores (the quantity the CI describes). Repeats share subjects, so their spread reflects split variability, not independent replication. Per-fold metrics are not plotted: a fold has 4–5 test subjects. |
| PLV vs \|ciPLV\| | Always separate conditions: colour, marker shape and filled vs open markers. PLV = blue filled circle, \|ciPLV\| = vermillion open square. Values are never averaged together. Signed ciPLV is never plotted as connectivity. |
| Significance | Never computed by the plotting code. Stars appear only if a statistics CSV is supplied whose rows are subject-level and carry a multiple-comparison-corrected p (`p_corrected`) and `correction_method` (see `visualization/stats.py`). The figure's `_meta.json` then says `kind: inferential`; otherwise `kind: descriptive`. |
| Thresholds | Classification thresholds are fixed (0.5 probability, 0 SVM margin). Network plots use a display-only edge selection, labelled as such in the figure, its data and its metadata. |
| Provenance | Each figure is written with its exact plotted data (`<id>_data.csv`/`.json`) and `<id>_meta.json`. Each run writes `figure_manifest.json` with config, style, source files (sha256), timestamp and git commit. Existing files are never overwritten. |
| Style | Styling is central (`configs/figures/style.yaml`): journal widths (single 85 mm, double 178 mm), font sizes, line widths, marker sizes. Output is PDF/SVG (editable text) plus PNG at 600 dpi. The palette is Okabe–Ito (colour-blind safe); categories also differ by marker and fill (grayscale). Maps use `cividis` (sequential) and `RdBu_r` (diverging, symmetric limits). Diverging sign is not recoverable in grayscale, so the data files carry the values. |

## Required input files

| Symbol | File (per experiment result directory `<exp>/<variant>/<setting>/`) |
|---|---|
| **M** | `metrics_repeats.csv`: subject-level metrics per repeat (scope `all`, plus `D1`/`D2` for pooled) |
| **Σ** | `summary.json`: repeat-averaged subject-level metrics and subject-bootstrap CI |
| **P** | `predictions_subjects.csv`: subject score per repeat |
| **W** | `predictions_windows.csv`: test-window scores and learned band weights (fusion figures only) |
| **F** | `folds.csv`: subject→fold assignment (paired comparisons require identical folds) |
| **H** | `hparams.csv`: `n_parameters`, `fit_seconds` per split |
| **E** | `embeddings.npz`: fused graph embeddings (runs with `evaluation.save_embeddings: true`) |
| **G** | graph run `graphs.npz` + `windows.csv` (`psd_log`, `psd_rel`, `plv`, `ciplv`, `roi_coords`) |
| **S** | optional statistics CSV (corrected, subject-level) |

---

## Family 1: experiment performance comparison

**F1_performance_within** (main), **S1_performance_pooled** (supp.),
**S2/S3_performance_pooled_D1/D2_subjects** (supp.), **S4_performance_cross** (supp.)

- **Question:** how do Exp 0–6 compare in subject-level classification, per evaluation setting?
- **Inputs:** M (S4 also Σ); S optional.
- **Axes:**
  - x = experiment/model entry.
  - y = metric (balanced accuracy, ROC-AUC, F1, sensitivity, specificity), one row per metric.
  - Columns = settings.
  - Conditions PLV / \|ciPLV\| / no connectivity side by side.
  - S2/S3 show pooled-model performance restricted to D1 or D2 subjects.
- **Unit:** subject-level metric per CV repeat (S4: bootstrap CI over subjects).
- **Interpretation:** relative performance and its split variability on the same subjects and splits.
- **Not proof of:**
  - A model being better. Overlapping repeat distributions are not tests, and repeats are dependent.
  - Generalization beyond these cohorts.
  - Any significance, unless stars from a supplied corrected statistics file are shown.
- **Status:** F1 main; S1–S4 supplementary.
- **Author decision pending:** the Exp 2/3 band and the models in the entry list must be fixed in advance.

## Family 2: PLV vs |ciPLV| paired comparison

**S5_plv_vs_ciplv_paired** (supp.)

- **Question:** for the same model, band and subjects, does the result change between PLV and \|ciPLV\|?
- **Inputs:** M, F (identical folds verified; unmatched pairs are refused and listed in the metadata).
- **Axes:**
  - x = matched configuration (experiment + variant without the connectivity tag).
  - y = metric.
  - Lines join the PLV and \|ciPLV\| values of the same repeat.
- **Unit:** paired subject-level metric per repeat.
- **Interpretation:** sensitivity of the result to zero-lag (leakage-prone) coupling.
- **Not proof of:**
  - Leakage being the cause of a difference.
  - \|ciPLV\| being "better connectivity".
- **Status:** supplementary.

## Family 3: frequency-band comparison

**F3_band_comparison** (main), **S6_band_comparison_all_encoders** (supp.)

- **Question:** which band's single-band graph carries more discriminative information:
  - with connectivity only (Exp 2);
  - with connectivity + matching-band PSD (Exp 3)?
- **Inputs:** M.
- **Axes:**
  - x = band (theta, alpha1, alpha2, beta).
  - y = ROC-AUC.
  - Rows = experiment/encoder; columns = setting; PLV / \|ciPLV\| as conditions.
- **Unit:** subject-level metric per repeat.
- **Interpretation:** band-wise descriptive comparison.
- **Not proof of:**
  - A band-specific disease mechanism.
  - Differences between bands being significant.
  - Bands being independent (they share subjects and splits).
- **Status:** F3 main (GCN); S6 supplementary (EdgeWeightedGAT, WeightedGraphSAGE, named as such).

## Family 4: Exp 6 ablations

**F4_ablation_exp6** (main), **S7_ablation_exp6_pooled_cross** (supp.)

- **Question:** which components of the Exp 6 model contribute? The components are PSD,
  strength, learned fusion and each band.
- **Inputs:** M, F (Δ only between results with identical folds).
- **Axes:**
  - x = full, no_psd, no_strength, no_learned_fusion, drop_theta … drop_beta.
  - Upper rows: metric.
  - Lower rows: paired Δ vs full, per repeat.
  - Separate rows for PLV and \|ciPLV\|; columns = setting.
- **Unit:** subject-level metric per repeat; Δ paired by repeat.
- **Interpretation:** a descriptive contribution analysis.
- **Not proof of:**
  - Causal importance of a component.
  - Significance of Δ.
  - Additivity of components.
- **Status:** F4 main; S7 supplementary.

## Family 5: source-space PSD

**F6_psd_relative_heatmap** (main); supplementary:
**S11_psd_log_heatmap**, **S12a/b_psd_rel_heatmap_D1/D2**, **S13a/b_psd_lobes_rel/log**

- **Question:** how does source-space band power differ descriptively between AD and HC
  across ROIs and bands?
- **Inputs:** G (`psd_rel` or `psd_log`); S optional (`psd:<kind>:<band>:<roi>`).
- **Axes:**
  - Heatmaps: rows = 68 ROIs (lh→rh, by lobe), columns = bands. Panels: AD mean, HC mean, AD − HC.
  - Lobe plots: lobe × group, with points = subjects and bars = SD across subjects.
- **Unit:** subject (windows averaged within subject first).
- **Interpretation:** descriptive spatial-spectral pattern of group means.
- **Not proof of:**
  - Group differences (no tests unless a corrected statistics file is supplied).
  - Absolute source amplitudes (template anatomy; log-absolute power is in arbitrary units).
  - Anything about D1 and D2 separately when the pooled view is shown.
- **Status:** F6 main; S11–S13 supplementary.

## Family 6: connectivity

**F7_plv_matrices** (main), **S14_ciplv_matrices**, **S15a/b_plv/ciplv_network_diff** (supp.)

- **Question:** what do group-average PLV / \|ciPLV\| networks and their AD − HC differences look like per band?
- **Inputs:** G (`plv` / `ciplv`; network views also `roi_coords`).
- **Axes:**
  - Matrices: 68 × 68, ordered lh→rh by lobe. Panels: AD mean, HC mean, AD − HC; rows = band.
  - Network: axial projection of ROI centroids (MNI x, y).
  - Network edges: the top-N |AD − HC| edges per band. This is a **display-only selection**.
  - Line width ∝ |value|; solid = positive, dashed = negative.
- **Unit:** subject (windows averaged within subject).
- **Interpretation:** a descriptive overview.
- **Not proof of:**
  - Altered connections. The displayed edges are selected by magnitude on the full labelled
    data, purely for visibility; they are not a statistical threshold.
  - Direction of influence (|ciPLV| is a magnitude).
- **Status:** F7 main; S14–S15 supplementary.

## Family 7: multi-band fusion weights

**F8_fusion_weights** (main)

- **Question:** how did the trained multi-band models weight the four band embeddings?
- **Inputs:** W (`band_weight_<band>`, test windows) for models with `gate`/`attention`
  fusion; S required for any AD vs HC split.
- **Axes:**
  - x = band; y = learned fusion weight.
  - Points = subjects; boxes = distribution across subjects.
  - Dotted line = uniform 1/K.
  - Panels = model × condition.
- **Unit:** subject (window weights averaged per subject and repeat, then over repeats).
- **Interpretation:** how the fitted model combines bands.
- **Not proof of:**
  - Biological importance of a band.
  - A group difference. The AD vs HC split is refused unless the statistics file supplies
    corrected results for every band.
- **Status:** main.

## Family 8: subject-level ROC

**F5a/F5b_roc_within_D1/D2** (main), **S10_roc_pooled** (supp.)

- **Question:** how well do the primary models rank AD above HC subjects?
- **Inputs:** P (Σ for cross-check).
- **Axes:**
  - FPR vs TPR from one repeat-averaged score per subject.
  - AUC and n subjects in the legend.
  - Optional pointwise 95 % subject-bootstrap band.
- **Unit:** subject.
- **Interpretation:** discrimination of the primary models in that setting.
- **Not proof of:**
  - Clinical utility.
  - Calibration (SVM scores are margins).
  - Performance on new cohorts.
- **Status:** F5 main; S10 supplementary.
- **Author decision pending:** confirm the primary models.

## Family 9: confusion matrices

**S8a/S8b_confusion_within_D1/D2** (supp.)

- **Inputs:** P.
- **Axes:**
  - True × predicted (HC, AD).
  - Raw counts and row-normalized panels.
  - Threshold fixed on the score scale.
- **Unit:** subject (repeat-averaged score).
- **Interpretation:** the error pattern (sensitivity/specificity balance).
- **Not proof of:**
  - Performance at other thresholds.
  - Clinical decision quality.
- **Status:** supplementary.

## Family 10: cross-dataset generalization

**F2_generalization** (main)

- **Question:** how does performance change from within-dataset to pooled to cross-dataset
  transfer (D1→D2, D2→D1)?
- **Inputs:** M, Σ.
- **Axes:**
  - x = within D1, within D2, pooled, D1→D2, D2→D1 (cross-dataset shaded).
  - y = ROC-AUC, balanced accuracy.
  - Marker = metric of repeat-averaged subject scores; bars = 95 % subject-bootstrap CI.
  - Points = repeats or seeds.
  - Panels = primary models.
- **Unit:** subject.
- **Interpretation:** a visual summary of domain shift.
- **Not proof of:**
  - Generalization, unless the DESIGN.md §8 criteria are met.
  - Valid cross-dataset results, unless the §7 gate passed (enforced by the runner).
- **Status:** main.

## Family 11: subject prediction distributions

**S9a/S9b_subject_scores_within_D1/D2** (supp.)

- **Inputs:** P.
- **Axes:**
  - x = true group; y = subject score (probability or SVM margin).
  - One point per subject; bars = min–max over repeats.
  - Box = distribution; dashed line = threshold.
  - Optional anonymized labels S01… (the mapping is never saved).
- **Unit:** subject.
- **Interpretation:** spread, overlap and borderline subjects.
- **Not proof of:**
  - Individual diagnosis.
  - Window-level reliability.
- **Status:** supplementary.

## Family 12: age sensitivity

**F9_age_sensitivity** (main)

- **Question:** does EEG add information beyond age?
- **Inputs:** M for `baseline_age`, `<exp>`, `<exp>_plus_age`.
- **Axes:**
  - x = Age only / EEG only / EEG + age.
  - y = ROC-AUC.
  - Rows = EEG model; columns = setting (by dataset).
- **Unit:** subject-level metric per repeat.
- **Interpretation:** whether EEG performance is matched by age alone, especially in D2,
  where HC ≈ 62 y and AD ≈ 80 y.
- **Not proof of:**
  - Absence of age confounding when EEG + age ≈ EEG only.
  - Age-independence of the EEG features.
- **Status:** main.

## Family 13: dataset shift (exploratory)

**S16_psd_pca_dataset_shift**, **S17_graph_embedding_pca** (supp.)

- **Question:** do D1 and D2 (and AD/HC) occupy different regions of feature or embedding space?
- **Inputs:**
  - S16: G (subject-mean `psd_log` + `psd_rel`, z-scored).
  - S17: E from one fitted model (one repeat/fold; embedding spaces of different folds are
    not comparable).
- **Axes:**
  - PC1/PC2 (variance explained in the labels); UMAP is optional.
  - Colour = diagnosis; marker = dataset.
  - S17: filled = training, open = test subjects.
- **Unit:** subject.
- **Interpretation:** descriptive, exploratory view of dataset shift.
- **Not proof of:**
  - Group or dataset differences. Visual clusters are not statistics.
  - The projection being fitted to separate groups (it is fitted without labels).
- **Status:** supplementary.

## Family 14: model complexity

**S18_complexity_params**, **S19_complexity_runtime** (supp.)

- **Inputs:** H, M.
- **Axes:**
  - x = mean trainable parameters (or fit time per split), log scale.
  - y = mean ± SD of the metric across repeats.
- **Unit:** subject-level metric per repeat; complexity per fitted model.
- **Interpretation:** the cost/benefit trade-off.
- **Not proof of:**
  - Efficiency on other hardware (runtime depends on the machine).
  - Better models having more parameters.
- **Status:** supplementary.

---

## Planned inventory

| ID | Family | Status | Inputs | Unit |
|---|---|---|---|---|
| F1_performance_within | 1 | main | M (+S) | subject metric / repeat |
| F2_generalization | 10 | main | M, Σ | subject (bootstrap CI) |
| F3_band_comparison | 3 | main | M | subject metric / repeat |
| F4_ablation_exp6 | 4 | main | M, F | subject metric / repeat (paired Δ) |
| F5a_roc_within_D1 / F5b_roc_within_D2 | 8 | main | P | subject |
| F6_psd_relative_heatmap | 5 | main | G (+S) | subject |
| F7_plv_matrices | 6 | main | G | subject |
| F8_fusion_weights | 7 | main | W (+S for group split) | subject |
| F9_age_sensitivity | 12 | main | M | subject metric / repeat |
| S1_performance_pooled | 1 | supp. | M | subject metric / repeat |
| S2/S3_performance_pooled_D1/D2_subjects | 1 | supp. | M (scope D1/D2) | subject metric / repeat |
| S4_performance_cross | 1 | supp. | M, Σ | subject (bootstrap CI) |
| S5_plv_vs_ciplv_paired | 2 | supp. | M, F | paired subject metric / repeat |
| S6_band_comparison_all_encoders | 3 | supp. | M | subject metric / repeat |
| S7_ablation_exp6_pooled_cross | 4 | supp. | M, F | subject metric / repeat |
| S8a/S8b_confusion_within_D1/D2 | 9 | supp. | P | subject |
| S9a/S9b_subject_scores_within_D1/D2 | 11 | supp. | P | subject |
| S10_roc_pooled | 8 | supp. | P | subject |
| S11_psd_log_heatmap | 5 | supp. | G | subject |
| S12a/S12b_psd_rel_heatmap_D1/D2 | 5 | supp. | G | subject |
| S13a/S13b_psd_lobes_rel/log | 5 | supp. | G | subject |
| S14_ciplv_matrices | 6 | supp. | G | subject |
| S15a/S15b_plv/ciplv_network_diff | 6 | supp. | G (`roi_coords`) | subject; display-only edges |
| S16_psd_pca_dataset_shift | 13 | supp. | G | subject (exploratory) |
| S17_graph_embedding_pca | 13 | supp. | E | subject (exploratory) |
| S18/S19_complexity_params/runtime | 14 | supp. | H, M | per model |

Every experiment output named above is produced by the existing runner; nothing has been
run on real data yet. Two inputs need non-default settings:
- `embeddings.npz` requires `evaluation.save_embeddings: true` for the Exp 6 pooled run.
- `roi_coords` is computed from fsaverage by graph runs made with the current pipeline.
