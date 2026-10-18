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

## 4. Result (provisional: sphere head model)

The fsaverage BEM of the stores cannot be downloaded in the environment where this release was prepared, so the
analysis was run with a three-shell sphere centred on the electrodes (an approximation; `02_parcellation` reruns it with
the stores' BEM and writes the table that counts). Leadfield: numerical rank **18**, entropy-effective rank **13.1**,
10 singular values hold 95 % and 15 hold 99 % of the energy, and only **5** lie above the wMNE regularization level.

| atlas | inverse | ROI-operator rank | redundant ROI signals | median cross-talk | ROIs dominated by own region | median \|leakage corr\| | participation ratio |
|---|---|---|---|---|---|---|---|
| dk68 | wMNE | 18 | 50 | 0.84 | 0.54 | 0.19 | 8.7 |
| dk68 | LCMV | 18 | 50 | 0.86 | 0.59 | 0.22 | 7.6 |
| dk34_gyral | wMNE | 18 | 16 | 0.71 | 0.71 | 0.21 | 6.8 |
| dk34_gyral | LCMV | 18 | 16 | 0.74 | 0.68 | 0.24 | 6.1 |
| r18 | wMNE | 18 | 0 | 0.48 | 0.89 | 0.24 | 5.7 |
| r18 | LCMV | 18 | 0 | 0.52 | 0.89 | 0.26 | 5.1 |
| lobes12 | wMNE | 12 | 0 | 0.38 | 1.00 | 0.29 | 5.1 |
| lobes12 | LCMV | 12 | 0 | 0.43 | 1.00 | 0.30 | 4.5 |

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

**What this needs (author approval):** r18 stores for LCMV and wMNE, built with the unchanged pipeline
(`scripts/build_stores.py build --inverse lcmv --atlas r18`, then wMNE, `validate`, `match`; CPU, about 40 min for
LCMV and 30-45 min for wMNE, measured on the earlier 68-ROI builds), followed by setting `release.PRIMARY["atlas"]`
to `r18` with a logged decision. The validated 68-ROI stores are never overwritten (r18 stores have their own names);
**dk68 remains the reference / sensitivity analysis** and the comparison with all earlier results. If the rebuild is
not approved, dk68 stays primary and the paper states the leakage limitation quantified in §4.

## 6. Why the saved 68-ROI data cannot be averaged into coarser regions

The stores save, per participant, the continuous 68 ROI time series (`roi_timeseries.npz`) and, for LCMV, the 68 x 19
ROI operator. A coarse region's signal is the `mean_flip` average over ALL sources of the merged label, with sign
flips derived from the merged label's own cortical normals. Each saved DK signal already carries its own sign flip,
whose sign relative to its neighbours is arbitrary, and its own source-count weighting; averaging DK signals (or their
operator rows) is therefore not the merged-label signal and can cancel activity, and averaging 68 x 68 PLV matrices is
not the connectivity of the merged regions (phase synchrony does not average). The legitimate route recomputes the ROI
operator for the merged labels from the forward model and the inverse (for LCMV from each participant's own data
covariance, i.e. from the recordings) and then the features and connectivity: exactly what the store pipeline does
with `source.atlas` set (`atlas.merged_labels`). Nothing in the final framework aggregates 68-ROI signals or
connectivity.

## 7. Infrastructure

`configs/source_graphs.yaml` + `resolve(inverse, atlas)` (atlas-specific store names, e.g.
`D3_LCMV_XU/graphsLX-r18_v1_<hash>`), merged labels in the store builder, atlas-aware ROI names, contiguity tables
(`source/atlas_tables.json`, `scripts/make_atlas_tables.py`), atlas-relative k for kNN graphs, lobes of every atlas for
the Exp 8 anatomical layer (the 12-lobe atlas with `lobe_hemisphere` grouping has no within-group pairs and is refused
for that layer), store verification per atlas.
