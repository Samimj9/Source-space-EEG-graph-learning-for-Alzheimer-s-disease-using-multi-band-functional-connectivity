# Cortical parcellation for 19-channel EEG

Code `adgraph.source.atlas` (atlases), `adgraph.source.resolution` (analysis), `scripts/final.py parcellation |
reliability`; notebook `02_parcellation`; rule `configs/final/selection.yaml` (`atlas_resolution`). Methodology log
entry 58. Tests `tests/test_resolution.py`.

## 1. The question

The stores reconstruct 68 Desikan-Killiany ROIs from 19 scalp electrodes. After the average reference the 19 channels
span at most 18 spatial dimensions, so 68 ROI signals are necessarily linear combinations of at most 18 independent
signals. Whether 68 ROIs are an appropriate resolution is therefore a property of the measurement, independent of any
classification result, and it is answered here without EEG data or labels.

## 2. Candidate atlases (`source.atlas`)

Every coarser region is a union of whole Desikan-Killiany labels of one hemisphere (never across hemispheres):
`dk68` (68), `dk34_gyral` (17 gyral groups per hemisphere), `r18` (9 large regions per hemisphere: prefrontal,
orbitofrontal, sensorimotor, parietal, lateral temporal, medial temporal, occipital, cingulate, insula), `lobes12`
(6 lobes per hemisphere). The number of regions is not tied to the number of electrodes; 18 is the rank bound.

## 3. Method

For each atlas and both inverses, the ROI operator W (R x 19) is built exactly as the stores build it (identity
"recording" through the MNE inverse and `mean_flip` label extraction; wMNE with the stores' depth 0.8, loose 0.2,
lambda2 1/9; LCMV with reg 0.05, unit-noise-gain-invariant weights and, because the beamformer is data-dependent, the
model covariance of independent unit-variance cortical sources plus noise at amplitude SNR 3). From the fsaverage
forward model and the region leadfield (unit activity of a region with the `mean_flip` sign pattern):

* spatial rank of the average-referenced normal-orientation leadfield (numerical, entropy-effective, energy);
* rank of W and the number of ROI signals forced to be linear combinations of the others;
* the region resolution matrix W G: cross-talk fraction per ROI (share of an ROI estimate's variance coming from
  other regions under independent equal-variance region activity) and own-region dominance;
* the leakage correlation between ROI estimates (the zero-lag correlation volume conduction alone puts into PLV-type
  measures; |ciPLV| discards the zero-lag part) and its participation ratio (effective number of independent signals).

The data-dependent parts (development participants only, label-blind; `scripts/final.py reliability`) add, for the
existing 68-ROI stores: the rank of every participant's saved LCMV ROI operator and the split-half (odd vs even
windows) reliability, across participants, of PSD / DE, node strength and edge-wise PLV / |ciPLV|.

## 4. Result (approximate 3-layer BEM; sphere model as a check)

The stores' fsaverage 5120 BEM could not be downloaded where this release was prepared. The analysis was therefore
run with a 3-layer BEM built from the two realistic fsaverage surfaces shipped with MNE (scalp and inner skull,
ico-4; outer skull a 3.3-5 mm radial expansion of the inner skull; `resolution.approximate_bem`), and repeated with a
three-shell sphere centred on the electrodes. The two agree within about 0.02 on every quantity below, so the
conclusion does not hinge on the head model; `02_parcellation` reruns it with the stores' exact BEM (in the Colab
cache) and writes the table that counts. Leadfield: numerical rank **18**, entropy-effective rank **13.3**, 10
singular values hold 95 % and 15 hold 99 % of the energy, and only **5** lie above the wMNE regularization level.

| atlas | inverse | ROI-operator rank | redundant ROI signals | median cross-talk | ROIs dominated by own region | median \|leakage corr\| | participation ratio |
|---|---|---|---|---|---|---|---|
| dk68 | wMNE | 18 | 50 | 0.85 | 0.53 | 0.19 | 8.6 |
| dk68 | LCMV | 18 | 50 | 0.86 | 0.57 | 0.23 | 7.4 |
| dk34_gyral | wMNE | 18 | 16 | 0.71 | 0.74 | 0.22 | 6.6 |
| dk34_gyral | LCMV | 18 | 16 | 0.72 | 0.74 | 0.26 | 5.8 |
| r18 | wMNE | 18 | 0 | 0.49 | 0.89 | 0.24 | 5.4 |
| r18 | LCMV | 18 | 0 | 0.53 | 0.89 | 0.27 | 4.8 |
| lobes12 | wMNE | 12 | 0 | 0.39 | 1.00 | 0.29 | 4.8 |
| lobes12 | LCMV | 12 | 0 | 0.46 | 1.00 | 0.30 | 4.3 |

Limitations: fsaverage template anatomy and template (Colin27 lookup) electrode positions, as in the stores; the
LCMV operator uses a model covariance (independent unit-variance sources), whereas each store uses the participant's
data covariance (the saved per-participant operators all have rank 18, `reliability`); resolution under independent
equal-variance region activity, not under real correlated cortical activity.

Reading: with 68 ROIs, 50 of the 68 signals are exact linear combinations of the others, the median ROI receives
about 85 % of its variance from other regions, and only about half of the ROIs are dominated by their own region:
most of the 68 x 68 connectivity entries are dominated by volume conduction and source leakage, not by distinct
cortical interactions. The 34-region atlas still carries 16 redundant signals. At 18 regions the forced redundancy
disappears, cross-talk halves and 89 % of the ROIs are dominated by their own region; 12 lobes separate best but are
anatomically coarse. Coarser regions have slightly higher pairwise leakage correlations (large neighbouring regions),
which |ciPLV| is designed to suppress.

## 5. Recommendation

**r18** (18 regions, 9 per hemisphere) is the finest atlas meeting every pre-specified criterion for both inverses
(no forced redundancy, median cross-talk <= 0.60, own-region dominance >= 0.80). The two numeric thresholds were set
after the provisional resolution numbers above were seen (no EEG data or diagnosis involved); the choice does not
depend on them: the threshold-free criterion "no more regions than the ROI operator's rank" alone excludes dk68 and
dk34 and makes r18 the finest admissible atlas. Literature (checked 2026-10-18; verify page details before citing):

* Farahibozorg, Henson & Hauk (2018), *NeuroImage* 169:23-45: parcellations should be adapted to the resolution of
  the source estimates, using cross-talk to merge leaky parcels; for their high-density combined MEG / EEG
  configuration the resolution limit was about 70 parcels, i.e. a 19-electrode EEG supports far fewer.
* Hauk, Stenroos & Treder (2022), *NeuroImage* 255:119177: resolution matrices, point-spread and cross-talk
  functions as the objective evaluation of linear estimators (minimum norm and LCMV), the approach used in §3.
* Palva et al. (2018), *NeuroImage* 173:632-643: field spread creates spurious ("ghost") interactions in source space
  even for measures insensitive to zero-lag coupling, so leakage limits connectivity estimates beyond what |ciPLV|
  removes.
* Song et al. (2015), *J Neurosci Methods* 256:9-21: source localization accuracy improves with sensor density and
  inferior head coverage, both limited with the 10-20 montage.

**Approved (author, 2026-10-18), conditional:** r18 stores for LCMV and wMNE, modes B and A, built from the original
recordings with the unchanged pipeline in notebook 02, only if the anatomical checks pass and r18 meets the rule with
the stores' exact BEM (`scripts/build_stores.py build | validate | match | match-atlas | size --atlas r18`; CPU,
about 40 min for LCMV and 30-45 min for wMNE, measured on the earlier 68-ROI builds). The validated 68-ROI stores are
never overwritten (r18 stores have their own names); **dk68 remains the reference / sensitivity analysis** and the
comparison with all earlier results. r18 is NOT made the analysis atlas by code: after the development-only
comparison of notebook 04 the author records the atlas (`final.py choose-atlas`). If dk68 is kept, the paper states
the leakage limitation quantified in §4.

**Caveat.** A rank-18 sensor space does not prove that 18 anatomical regions are independently or accurately
localized: a median r18 estimate still receives about half of its (independent, equal-variance) variance from other
regions, and real cortical activity is neither independent nor equal in variance.

## 6. Can coarser regions be obtained from the saved 68-ROI data?

**Not by averaging.** A coarse region's signal is the `mean_flip` average over ALL sources of the merged label, with
sign flips derived from the merged label's own cortical normals. Each saved DK signal carries its own sign flip, whose
sign relative to its neighbours is arbitrary, and its own source-count weighting; averaging DK signals (or their
operator rows) is not the merged-label signal and can cancel activity, and averaging 68 x 68 PLV matrices is not the
connectivity of the merged regions (phase synchrony does not average). Nothing in the final framework aggregates
68-ROI signals or connectivity.

**What the saved intermediates do allow** (`resolution.sensor_signals`, `resolution.reconstruction_check`,
`scripts/final.py reconstruct-check`, tested). Each store saves, per participant, the 68 x 19 ROI operator W
(`operator.npz`, both inverses) and the continuous ROI signals of every analysed piece (`roi_timeseries.npz`, float32,
without the analytic-filter margins). W rejects the constant vector and has rank 18, so its row space is exactly the
average-referenced sensor space: the sensor signals of each saved piece are recovered exactly as pinv(W) * roi (up to
float32). From them, a merged-label operator (forward model + inverse; for LCMV the data covariance of the accepted
samples, which lie inside the pieces) and coarse ROI signals could be computed without the recordings. Two limits:

* windows whose filter support reaches into the unsaved margins (within 825 samples = 3.3 s of a piece end, i.e. the
  first and last few windows of every piece) cannot be recomputed exactly; the real fraction per
  participant is measured by `reconstruct-check` in notebook 02 (3b);
* the result would be a second, differently built store family (a new code path, its own validation), not the
  unchanged store pipeline.

**Recommendation:** use the unchanged store pipeline with `source.atlas` set (`atlas.merged_labels`) on the
recordings, which reproduces every window exactly and is validated end to end by the existing `validate` / `match`
steps. The shortcut saves about 1 h of CPU and is not worth a second code path; it is documented here because it
shows that the coarse stores need no new data, only recomputation.

## 7. The r18 workflow (notebooks 02 and 04)

1. **Anatomy** (`atlas.verify_atlas`, `final.py atlas-check --atlas r18`): per hemisphere a partition of the 34 DK
   labels, no cross-hemisphere region, identical left / right members, every region one contiguous piece of cortex,
   centroids on the correct side, non-empty source sets. Mixed-lobe regions are reported (`sensorimotor`: frontal +
   parietal by design). Sizes: 321 (insula) to 2259 (parietal) ico-5 sources per region.
2. **Analytic resolution with the stores' BEM** (`final.py parcellation`; §3-4) and **the stores' own operators**
   (`final.py store-resolution`): every development participant's saved operator against the region leadfield of the
   store's cached forward model (verified to reproduce the analytic metrics for identical operators). The corrected
   LCMV store has no checkpoints of its own: its operators are read from the original build after provenance
   verification (`sourcegraphs.checkpoints`; the correction recomputed no operator; D3-23 and D3-58 carry the
   covariance caveat of methodology log entry 53). Remaining
   approximations: fsaverage anatomy, template (Colin27 lookup) electrode positions, cross-talk under independent
   equal-variance region activity.
3. **Stores** from the recordings (`build_stores.py build`), `validate`, `match` (wMNE vs LCMV r18), `match-atlas`
   (each r18 store vs its dk68 store: configuration identical except `store_names` / `source.atlas`; identical
   participants, labels, selections, windows, split, bands, channels, sampling, sensor features; experiments refuse
   an r18 store without it), `size`. The r18 LCMV covariance uses the corrected window selection; the dk68 `_corr1`
   LCMV covariance of D3-23 and D3-58 included one extra clean window (methodology log entry 53).
   Status (2026-10-19): with the stores' exact BEM, r18 LCMV own-region dominance is 0.778 < 0.80, so r18 does not
   meet the rule and notebook 02 stops before this step; the rule recommends lobes12 (methodology log entry 59a).
   No store is built and no atlas is changed without the author's decision.
4. **QC** of the four r18 stores (`final.py qc --atlas r18`), then the resolution and reliability of the r18
   operators actually built.
5. **Controlled comparison** (notebook 04, `configs/final/roi_comparison.yaml`), development CV only; the author
   records the atlas.

## 8. Infrastructure

`configs/source_graphs.yaml` + `resolve(inverse, atlas)` (atlas-specific store names, e.g.
`D3_LCMV_XU/graphsLX-r18_v1_<hash>`), merged labels in the store builder, atlas-aware ROI names, contiguity tables
(`source/atlas_tables.json`, `scripts/make_atlas_tables.py`), atlas-relative k for kNN graphs, lobes of every atlas for
the Exp 8 anatomical layer (the 12-lobe atlas with `lobe_hemisphere` grouping has no within-group pairs and is refused
for that layer), store verification per atlas.
