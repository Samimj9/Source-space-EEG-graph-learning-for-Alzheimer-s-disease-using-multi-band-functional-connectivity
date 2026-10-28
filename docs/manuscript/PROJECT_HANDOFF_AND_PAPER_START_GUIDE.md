# D3 source-space EEG-GNN project: handoff and "how to start the paper" guide

Written 2026-10-28 from the working session. Everything below was verified against saved files, code or a web search during the session unless it is marked **(not verified)**.

---

## 1. What the project is

Source-space EEG graph learning for Alzheimer's disease (AD) versus healthy controls (HC) on OpenNeuro **ds004504**: 65 participants (36 AD, 29 HC; FTD excluded by design), 19 scalp electrodes, resting state, eyes closed.

* Frozen participant split: **52 development** (29 AD / 23 HC) and **13 holdout** (7 AD / 6 HC); seed 20250702.
* Development evaluation: 5-fold participant-grouped cross-validation, one score per participant (mean of its window scores), threshold 0.5, primary metric participant-level ROC-AUC.
* Pipeline: boundary-aware segmentation, 250 Hz, average reference, 10 s windows with 1 s step, five-minute selection (mode B; mode A = first 10 windows), LCMV or wMNE sources into 68 Desikan-Killiany ROIs, merged to r18 / lobes12, five bands, PLV or |ciPLV|, PSD / DE / strength node features, GNNs trained for exactly 60 epochs.

## 2. What was done in this chat (chronological)

1. **Amendment A3** (branch `claude/a3-final-analysis`): dual-namespace freeze of the final analysis, holdout orchestration, consolidated development report, statistics, D3-to-D2 manifest, notebooks, tests. Drive-compatible write-once, numerical-environment policy (Exp6 ran on a CPU torch build, the rest on CUDA), single-model holdout path. You ran the freeze, the dev report and the single-model holdout yourself on Drive; I never ran a real-data freeze or holdout.
2. **Research synthesis** (43 pages): `docs/final/D3_Research_Synthesis_Source_Space_EEG_GNN_AD.docx`, reconstructed from saved outputs. Exp0/Exp1 (classical baselines) are excluded by instruction.
3. **Manuscript package** (this commit series): OJEMB-style manuscript draft v1, supplementary material, figure plan, existing-figure audit, reproducible figure builder, Colab notebook, integrity checklist.

## 3. Key verified results (development CV, n = 52, mode B unless stated)

| Model | Inverse | Atlas | AUC (DeLong 95% CI) | Acc | BA | Sens | Spec | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| **Primary: Exp8a dual-layer GCN, `plv__psd`** (retrospective emphasis) | LCMV | lobes12 | 0.8576 (0.749–0.966) | 0.846 | 0.858 | 0.759 | 0.957 | 22/1/7/22 |
| Exp5 `plv__separate__mean` (post-hoc atlas, A1) | LCMV | r18 | 0.8591 (0.743–0.975) | 0.865 | 0.866 | 0.862 | 0.870 | 20/3/4/25 |
| Exp5 same configuration | LCMV | lobes12 / dk68 | 0.847 / 0.841 | 0.788 / 0.846 | | | | |
| Exp4 band-mean GCN reference | LCMV | lobes12 | 0.844 (0.725–0.963) | 0.808 | 0.809 | 0.793 | 0.826 | 19/4/6/23 |
| Highest GNN AUC: Exp8a `plv__psd_de_strength__abl_anatomical_both` | wMNE | lobes12 | 0.873 (0.767–0.978) | 0.769 | 0.766 | 0.793 | 0.739 | |
| Its **descriptive holdout** (13 participants, previously inspected) | wMNE | lobes12 | 0.786 (0.522–1.000) | 0.615 | 0.631 | 0.429 | 0.833 | 5/1/4/3 |

* The primary model's BA equals its AUC to three decimals (0.8576): a numerical coincidence, verified from the confusion matrix and score file.
* The holdout result belongs to the wMNE anatomical-both model only, **never** to the LCMV primary model.
* Atlas physics (exact BEM): leadfield numerical rank 18, effective rank 12.8. Rule (median cross-talk ≤ 0.60, own-region dominance ≥ 0.80, no forced redundancy) rejects dk68, dk34_gyral (analytic only) and r18 (LCMV dominance 0.778) and accepts only lobes12 (0.415 / 0.833). No 20-region atlas exists.
* Paired tests: after Holm correction no difference among architectures, PLV vs |ciPLV|, LCMV vs wMNE, anatomical-layer ablations or atlases; mode B vs A has one supported pair (Exp7).
* Selection optimism of the best of 52 GNN candidates: +0.075 (LCMV) / +0.058 (wMNE); out-of-bag winner AUC 0.808 / 0.822.
* Counts: 320 saved results = 280 GNN + 40 classical; 104 GNN units in the frozen plan (116 including classical).

## 4. Rules that must stay in the paper

* Development cross-validation is not independent validation; the 13-person holdout was inspected earlier and is descriptive only. No D2 / external result exists.
* The choice of Exp8a LCMV `plv__psd` as the central model is **retrospective**, not pre-specified.
* Do not claim the anatomical layer helps (ablations show no supported benefit); the anatomical adjacency is atlas-derived cortical-surface contiguity, not structural connectivity.
* Exp8 (original, failed on lobes12 because the layer was the zero matrix) and Exp8a (amendment A2) are different things.
* Keep the r18 result prominent but disclose its chronology (approved after the exact-BEM result).
* Do not claim superiority over published work or over the classical baselines; do not claim clinical readiness.
* Never commit data (the repo ignores CSV and data files); never modify frozen results, pins or the freeze.

## 5. Material problems found (resolve before submission)

1. The Markdown handoff file named in the request was not available to the session, and only one of the three literature workbooks was supplied.
2. Saved comparison families LCMV-vs-wMNE (124) and mode B-vs-A (128) include Exp1/Exp0 pairs; GNN-only they are 116 pairs each (medians +0.021 and +0.039). Holm p-values were computed over the full families.
3. "Selection-corrected AUC" is ambiguous (observed minus optimism 0.790 / 0.814 versus out-of-bag winner 0.808 / 0.822); the manuscript reports both.
4. The atlas **rule** is label-blind and pre-specified; the recorded atlas **decision** (`atlas.json`, 2026-10-19T15:32:56Z) followed the development-only comparison and cites classification performance. Confirm from git history that the rule thresholds predate the first resolution results.
5. Classical baselines were excluded from the paper by instruction. In the saved analysis their best observed AUC (0.897 LCMV, 0.918 wMNE) is above the best GNN (0.865, 0.873), with a selection-aware difference interval that includes zero. Decide whether to report them; reviewers will ask.
6. Existing Drive figures fig1, fig2, fig3, fig4 and fig6 contain Exp0/Exp1 data; use the regenerated versions.
7. The repo dataset note gives a different dataset-paper title than the published one.
8. The official OJEMB template was not downloaded; limits (about 3,000 words, at most 8 display items, abstract at most 150 words, at most 50 references, Visual Summary) come from journal-page search snippets and must be re-checked.

## 6. Instructions: how to start (and finish) the paper

1. **Read** `docs/manuscript/D3_Research_Integrity_and_Submission_Checklist.docx` first (sections 2, 3 and 5), then the manuscript draft.
2. **Get the official template**: open the OJEMB pages below, download the *Science Manuscript* Word or LaTeX template and note the current limits. Paste the draft text (`D3_IEEE_OJEMB_Science_Manuscript_v1.docx`) into it; convert the plain-text equations (1)–(12) to Word or LaTeX equations; fill in authors, affiliations, funding, ethics, data availability (dataset is public, CC0).
3. **Regenerate the figures** in Colab: open `manuscript_figures/D3_Manuscript_Figure_Generation.ipynb` (branch `claude/a3-final-analysis`), run all cells with Drive mounted at `AD-LEB/processed`. It validates inputs (n = 52, recomputed AUC equals saved AUC), writes PDF/SVG/600-dpi PNG plus `figure_manifest.csv` and `source_inventory.csv`. Inspect every figure. Optional: run `render_cortical_surface()` for a surface rendering of lobes12.
4. **Decide the open items** in section 5 above (items 2, 4 and 5 matter most). If you want GNN-only Holm correction, it is a read-only re-analysis from saved scores.
5. **Prepare the Visual Summary** (suggested from Fig. 1 and Fig. 3g) and move the supplementary material (`D3_Supplementary_Material.docx` + the 280-row table) into the journal's supplementary format.
6. **Verify references**: the flagged items are the Song et al. volume/pages, the Van Veen et al. author list, the Cao et al. journal name and the Ding et al. venue.
7. **Final pass**: re-read Results and Discussion against Section 4 of this file; make sure no sentence implies the anatomical layer helps, that the holdout describes the primary model, that results are external validation, or that graph models beat non-graph baselines.
8. Optional strengthening: a prospectively specified external evaluation (a D3-to-D2 manifest exists; nothing has been run), a sex/age sensitivity analysis, and a paired comparison against the classical baselines.

## 7. Files produced (repository, branch `claude/a3-final-analysis`)

| Item | Path |
|---|---|
| Manuscript draft v1 | `docs/manuscript/D3_IEEE_OJEMB_Science_Manuscript_v1.docx` |
| Supplementary material | `docs/manuscript/D3_Supplementary_Material.docx` |
| Figure plan and captions | `docs/manuscript/D3_Manuscript_Figures_and_Captions.docx` |
| Existing-figure audit | `docs/manuscript/D3_Existing_Figures_Audit.md` |
| Integrity and submission checklist | `docs/manuscript/D3_Research_Integrity_and_Submission_Checklist.docx` |
| 43-page synthesis | `docs/final/D3_Research_Synthesis_Source_Space_EEG_GNN_AD.docx` |
| Figure builder, notebook, captions, README | `manuscript_figures/` |
| Generated figures (PDF, SVG, PNG) | `manuscript_figures/generated/` |
| 280-row results table | `supplementary_all_280_gnn_configurations.csv` (sent in chat; not committed because the repo ignores CSV) |
| A3 design and single-model docs | `docs/final/a3_design.md`, `a3_single_model.md`, `a3_review_checklist.md`; methodology log entries 58–62 |

## 8. Links

**Code**
* Repository: https://github.com/Samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity
* Working branch: https://github.com/Samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity/tree/claude/a3-final-analysis
* Pins: release `59a9ab70a55d107c5ad72acff1fb48f3ef42dff0` (Exp0–7), amendment A2 `7fa470c668b8cce64ca152372422638be3ee8203` (Exp8a), A3 verified `e552e94e6050383834ad6339ad1a546fb16b1d4e`, synthesis commit `2e2c495`, manuscript commit `679a12f`.

**Google Drive** (Drive folder IDs; sign in with the project account)
* FINAL_v1 results root: https://drive.google.com/drive/folders/1AakVEMCN44ng2HsoA1k4cPesZkMyaAbc
* Consolidated development analysis (CSV tables, `selection_check.json`, `interpretation.md`): https://drive.google.com/drive/folders/16Dw5PqGlMA---Ty7gd2_rM6hnV6zQHn2
* Consolidated figures (fig1–fig7, PDF + PNG): https://drive.google.com/drive/folders/1X-r_c1RxJIuXy53-t2TjAjR15ho_lwq9
* Release results tree: https://drive.google.com/drive/folders/1z321nK1VfNUwkFKYBR_HbuFFcH_4Fbv0
* Amendments: https://drive.google.com/drive/folders/1jsunGvExqcypvCqoGybPztGJ528tXMen ; A2 (Exp8a): https://drive.google.com/drive/folders/1H0wBw-lRIlqc-w14Dz55myNRBFvJ9xWj ; A3 final: https://drive.google.com/drive/folders/1t8xtnL-tf06RmDy9rPlLmoIG23QYvakT
* Reports: https://drive.google.com/drive/folders/1kX-Vx2TPXCSEfJfksbr68wNfD7xpF4wK
* Selection check: https://drive.google.com/file/d/1lJYu522ramGJh8S5TUR-wT7X7PU5GOcA/view ; atlas decision: https://drive.google.com/file/d/1yLKdBjfIBaMmi1SSZbr2PyegCcYirPFu/view
* Primary model participant predictions: https://drive.google.com/file/d/1vdl0LuarpDh0o4skpBpgxz66CP3kaq5b/view ; Exp5 r18: https://drive.google.com/file/d/1iSo42jTfFTDWYHv6ikdLAoQkCBuIWYl_/view ; Exp4 reference: https://drive.google.com/file/d/1GwJb-tLj_7Xo5nNrZsFf8le26kkl3GIH/view
* Colab path used by the code: `/content/drive/MyDrive/AD-LEB/processed/FINAL_v1/...`

**Journal (OJEMB)**
* Instructions for Authors / manuscript formats: https://www.embs.org/ojemb/manuscript-formats/
* Templates: https://www.embs.org/ojemb/ieee-ojemb-templates/
* About: https://www.embs.org/ojemb/about-ojemb/

**Data and key references**
* Dataset OpenNeuro ds004504: https://openneuro.org/datasets/ds004504 (doi 10.18112/openneuro.ds004504.v1.0.9) and data descriptor https://doi.org/10.3390/data8060095
* Xu et al. 2025 (dual-mode connectivity MF-MGCN): https://doi.org/10.3389/fnins.2025.1555657 ; PMC copy https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263931/
* Cao et al. 2025: https://doi.org/10.1186/s12938-025-01361-0 ; Burger et al. 2025 (ECoGNet): https://doi.org/10.1109/IJCNN64981.2025.11228047 ; Ding et al.: https://arxiv.org/abs/2510.11917
* Bruña et al. 2018 (ciPLV): https://arxiv.org/abs/1710.08037 ; Kipf and Welling: https://arxiv.org/abs/1609.02907 ; Hassan and Wendling: https://arxiv.org/abs/1801.02549

## 9. Limits of this session
Web pages could not be fetched directly (DNS blocked), so the journal limits and literature numbers come from search snippets and the literature workbook; Drive paths for the live Colab run were not executed; per-experiment diagnostic figures were sampled, not exhaustively audited; no experiment, training or holdout was run.
