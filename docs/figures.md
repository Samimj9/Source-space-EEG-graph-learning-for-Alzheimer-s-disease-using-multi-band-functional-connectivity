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

## Pre-result design decisions (frozen 2026-10-12, before any real result)

These decisions are recorded in `docs/methodology_architecture_log.md`, entries 25–30.

1. **PLV is the primary connectivity analysis.**
   - |ciPLV| is the predefined robustness/sensitivity analysis.
   - The two are never combined: no averaging, and no picking the better one per model.
   - A direct PLV-vs-|ciPLV| comparison uses only results with identical subject folds and
     repeats. The paired figure verifies `folds.csv` and refuses unmatched pairs.
2. **No representative band for Exp 2 / Exp 3.**
   - They are band-specific by design and are compared only across theta, alpha1, alpha2 and
     beta (Fig. 4, S07, S08).
   - The overall comparison (Fig. 3) contains Exp 0, 1, 4, 5 and 6 only. This avoids
     choosing a band after seeing results.
3. **Main metrics:** balanced accuracy and ROC-AUC. Accuracy, F1, sensitivity and
   specificity remain available in supplementary figures (S01) and tables.
4. **Primary models:**
   - Exp 0 SVM: conventional sensor-PSD baseline.
   - Exp 1 SVM: source-PSD baseline.
   - Exp 6 full: proposed graph model.
   - They are the default entries for ROC (S10), confusion (S11), subject-score (S12) and
     generalization (Fig. 6) figures. Other models remain configurable.
5. **Main-paper figures are frozen** as listed below. Everything else is supplementary
   unless results later justify a promotion, which must be logged with its reason:
   ROC, confusion matrices, detailed |ciPLV| plots, subject-score distributions,
   PCA/embedding views, network views, complexity plots and the age sensitivity figure.
6. **Fit-time comparisons** only within one computational environment (see Family 14).
7. **Graph-embedding views:** one explicitly specified fitted model only (see Family 13).

### Main-paper figures

| Fig. | Content | Generated figure id(s) |
|---|---|---|
| 1 | Methodological pipeline (conceptual schematic) | none; drawn later, not data-derived |
| 2 | Source-space spectral / network characterization | `Fig2a_source_psd_relative`, `Fig2b_plv_group_matrices` |
| 3 | Overall performance: Exp 0 / 1 / 4 / 5 / 6 (balanced accuracy, ROC-AUC) | `Fig3_performance_overall` |
| 4 | Band-specific Exp 2 (connectivity only) vs Exp 3 (connectivity + matching-band PSD) | `Fig4_band_specific_exp2_vs_exp3` |
| 5 | Exp 6 ablations | `Fig5_exp6_ablations` |
| 6 | Within-dataset / pooled / cross-dataset generalization | `Fig6_generalization` |
| 7 | Multi-band interpretation: learned fusion weights | `Fig7_multiband_fusion` |

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

**Fig3_performance_overall** (main);
**S01_performance_overall_other_metrics**, **S02_performance_overall_pooled**,
**S03/S04_performance_pooled_D1/D2_subjects**, **S05_performance_overall_cross** (supp.)

- **Question:** how do Exp 0, 1, 4, 5 and 6 compare in subject-level classification, per evaluation setting? Exp 2/3 are excluded by design (band-specific; Fig. 4).
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
- **Metrics:**
  - Fig. 3: balanced accuracy and ROC-AUC.
  - S01: accuracy, F1, sensitivity, specificity.
- **Status:** Fig. 3 main; S01–S05 supplementary.

## Family 2: PLV vs |ciPLV| paired comparison

**S06_plv_vs_ciplv_paired** (supp.)

- **Question:** for the same model, band and subjects, does the result change between PLV (primary) and \|ciPLV\| (robustness)?
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

**Fig4_band_specific_exp2_vs_exp3** (main), **S07_band_specific_other_encoders**, **S08_band_specific_balanced_accuracy** (supp.)

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
- **Status:** Fig. 4 main (GCN, ROC-AUC). Supplementary: S07 (EdgeWeightedGAT, WeightedGraphSAGE, named as such) and S08 (balanced accuracy).
- **No representative band:** this is the only place Exp 2/3 are compared.

## Family 4: Exp 6 ablations

**Fig5_exp6_ablations** (main), **S09_exp6_ablations_pooled** (supp.)

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
- **Status:** Fig. 5 main; S09 supplementary.

## Family 5: source-space PSD

**Fig2a_source_psd_relative** (main); supplementary:
**S14_source_psd_log**, **S15a/b_source_psd_relative_D1/D2**, **S16a/b_source_psd_lobes_rel/log**

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
- **Status:** Fig. 2a main; S14–S16 supplementary.

## Family 6: connectivity

**Fig2b_plv_group_matrices** (main), **S17_ciplv_group_matrices**, **S18a/b_plv/ciplv_network_diff** (supp.)

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
- **Status:** Fig. 2b main (PLV, primary). |ciPLV| matrices (S17) and all network views (S18) are supplementary.

## Family 7: multi-band fusion weights

**Fig7_multiband_fusion** (main)

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

**S10a/b/c_roc_within_D1 / within_D2 / pooled** (supp.)

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
- **Models (default):** Exp 0 SVM, Exp 1 SVM, Exp 6 full. Others are configurable.
- **Status:** supplementary, unless promoted with a logged reason.

## Family 9: confusion matrices

**S11a/S11b_confusion_within_D1/D2** (supp.)

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

**Fig6_generalization** (main; primary models; balanced accuracy and ROC-AUC)

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

**S12a/S12b_subject_scores_within_D1/D2** (supp.)

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

**S13_age_sensitivity** (supp.)

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
- **Status:** supplementary (not in the frozen main set); may be promoted if results warrant,
  with a logged reason.

## Family 13: dataset shift (exploratory)

**S19_psd_pca_dataset_shift**, **S20_graph_embedding_pca** (supp.)

- **Question:** do D1 and D2 (and AD/HC) occupy different regions of feature or embedding space?
- **Inputs:**
  - S16: G (subject-mean `psd_log` + `psd_rel`, z-scored).
  - S20: E from **one explicitly specified fitted model**. Integer `repeat` and `fold` are
    required, otherwise the figure is refused.
  - Embeddings from independently trained folds live in unaligned coordinate systems and
    are never merged or co-plotted.
- **Axes:**
  - PC1/PC2 (variance explained in the labels); UMAP is optional.
  - Colour = diagnosis; marker = dataset.
  - S20: filled = training, open = test subjects.
- **Unit:** subject.
- **Interpretation:** descriptive, exploratory view of dataset shift.
- **Not proof of:**
  - Group or dataset differences. Visual clusters are not statistics.
  - The projection being fitted to separate groups (it is fitted without labels).
- **Status:** supplementary.

## Family 14: model complexity

**S21a_complexity_parameters**, **S21b_complexity_fit_time** (supp.)

- **Inputs:** H, M, and for fit time the `compute_environment` in each `manifest.json`.
  That record holds the device, CPU model and count, GPU, torch threads, platform, library
  versions and a fingerprint.
- **Comparability:**
  - Parameter counts are always reproducible.
  - Fit times are compared only if every result has the **same environment fingerprint**.
    Results with different or missing environments are refused (`IncomparableTiming`).
  - The environment is written into the figure metadata.
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
| Fig. 1 | pipeline schematic | main | none (conceptual, drawn later) | none |
| Fig2a_source_psd_relative | 5 | main | G (`psd_rel`) (+S) | subject |
| Fig2b_plv_group_matrices | 6 | main | G (`plv`) | subject |
| Fig3_performance_overall | 1 | main | M (+S) | subject metric / repeat |
| Fig4_band_specific_exp2_vs_exp3 | 3 | main | M | subject metric / repeat |
| Fig5_exp6_ablations | 4 | main | M, F | subject metric / repeat (paired Δ) |
| Fig6_generalization | 10 | main | M, Σ | subject (bootstrap CI) |
| Fig7_multiband_fusion | 7 | main | W (+S for group split) | subject |
| S01_performance_overall_other_metrics | 1 | supp. | M | subject metric / repeat |
| S02_performance_overall_pooled | 1 | supp. | M | subject metric / repeat |
| S03/S04_performance_pooled_D1/D2_subjects | 1 | supp. | M (scope D1/D2) | subject metric / repeat |
| S05_performance_overall_cross | 1 | supp. | M, Σ | subject (bootstrap CI) |
| S06_plv_vs_ciplv_paired | 2 | supp. | M, F | paired subject metric / repeat |
| S07_band_specific_other_encoders | 3 | supp. | M | subject metric / repeat |
| S08_band_specific_balanced_accuracy | 3 | supp. | M | subject metric / repeat |
| S09_exp6_ablations_pooled | 4 | supp. | M, F | subject metric / repeat |
| S10a/b/c_roc_within_D1/within_D2/pooled | 8 | supp. | P | subject |
| S11a/b_confusion_within_D1/D2 | 9 | supp. | P | subject |
| S12a/b_subject_scores_within_D1/D2 | 11 | supp. | P | subject |
| S13_age_sensitivity | 12 | supp. | M | subject metric / repeat |
| S14_source_psd_log | 5 | supp. | G | subject |
| S15a/b_source_psd_relative_D1/D2 | 5 | supp. | G | subject |
| S16a/b_source_psd_lobes_rel/log | 5 | supp. | G | subject |
| S17_ciplv_group_matrices | 6 | supp. | G | subject |
| S18a/b_plv/ciplv_network_diff | 6 | supp. | G (`roi_coords`) | subject; display-only edges |
| S19_psd_pca_dataset_shift | 13 | supp. | G | subject (exploratory) |
| S20_graph_embedding_pca | 13 | supp. | E (one fitted model) | subject (exploratory) |
| S21a_complexity_parameters | 14 | supp. | H, M | per model |
| S21b_complexity_fit_time | 14 | supp. | H, M, `compute_environment` (one environment) | per model |

Every experiment output named above is produced by the existing runner; nothing has been
run on real data yet. Two inputs need non-default settings:
- `embeddings.npz` requires `evaluation.save_embeddings: true` for the Exp 6 pooled run.
- `roi_coords` is computed from fsaverage by graph runs made with the current pipeline.
