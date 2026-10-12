# Data inventory

Snapshot of the `AD-LEB` Google Drive folder, taken 2026-10-12 (read-only,
via the Drive connector). No data from here is stored in the repository. This
file records structure, counts and issues only.

## Raw EEG

| Dataset | Folder | Files | Format | Channels (per manifest) | Used |
|---|---|---|---|---|---|
| D1 (2017) | `EEGs/Dataset 1 (2017)/` | 20 (`sub1`–`sub20.edf`) | EDF, ~24–39 MB | 32 incl. M1/M2 → 30 after mastoid drop | yes |
| D2 (2019) resting | `EEGs/Dataset 2 (2019)/RestingState/` | 23 (`<id>_<DDMYYYY>_1.edf`) | EDF, ~20–37 MB | 32 incl. M1/M2 | yes |
| D2 (2019) picture naming | `.../PictureNaming/` | `<id>_<date>_2.gdf` | GDF, ~170 MB | n/a | no |
| D2 (2019) button press | `.../ButtonPress/` | `<id>_<date>_3.gdf` | GDF, ~50–80 MB | n/a | no |

`electrodes_positions_32.loc`: EEGLAB polar layout (template, not digitized):
Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz C4 T8 M2 CP5 CP1 CP2 CP6
P7 P3 Pz P4 P8 POz O1 Oz O2.

## Participants

### Dataset 1 (`Info Participants_2017.xlsx`)
Columns: `ID, Age, Gender, Group, MMSE, Data Quality, Comments`.

- 20 subjects: **10 AD** (sub1, 3, 4, 5, 8, 9, 11, 17, 19, 20) and
  **10 HC** (sub2, 6, 7, 10, 12, 13, 14, 15, 16, 18).
- Mean age is about 71 for AD; HC is about the same once sub18 is excluded.
- No ACE-R.

### Dataset 2 (`Info Participants_2019.xlsx`)
Columns: `Id, Age, Gender, Group, ACE-R (/100), MMSE (/30)`, five ACE-R
subscores, `Data Quality, Comments`. `Id` looks like `2_842019`
(subject_date), and it matches the EDF filename prefix.

- 23 subjects: **13 HC** (2–7, 9–15) and **10 AD** (17–23, 26–28).
- Subject IDs 1, 8, 16, 24, 25 are absent. ID 24 has task recordings
  (PictureNaming/ButtonPress) but no resting-state file and no metadata row.

## Processed outputs (`processed/`)

| File | Date | Content |
|---|---|---|
| `run_manifest_20260912_162006.json` | 2026-09-12 | Full parameter record for the first graph run: 826 graphs, 43 subjects (D1 + D2) |
| `graph_summary_20260912_162006.csv` | 2026-09-12 | One row per graph (window) for that run |
| `graph_summary_20260915_141235.csv` | 2026-09-15 | D2-only rerun, 23 subjects; columns `subject_id, dataset, label, mmse, age, sex, comment, tmin, tmax, sample_weight` (`sample_weight = 1/n_windows_subject`) |
| `all_graphs_20260915_144715.pkl` | 2026-09-15 | ~124 MB pickled graph set. **No matching manifest**, so its contents and config have to be confirmed before use |

## Known issues (must be resolved or explicitly handled)

1. **D1 duplicate recordings with conflicting labels (blocking).** The
   manifest reports duplicate-content files. Fingerprint-confirmed:
   `sub1`(AD, MMSE 21) = `sub2`(HC, MMSE 27) and `sub13`(HC, MMSE 30) =
   `sub17`(AD, MMSE 9). Same byte size, not yet confirmed:
   `sub3`(AD)/`sub10`(HC) and `sub9`(AD)/`sub14`(HC). Every pair crosses the
   AD/HC boundary, so the same signal carries both labels. The manifest says
   "5 pairs" but names only 4. `sub6`/`sub7`/`sub8` also share one byte size,
   though identical EDF size can simply mean identical duration and channel
   count. Next steps: hash the signal arrays of all D1 files, then ask the
   data owner which label is correct, or exclude the affected subjects.
2. **D1 `sub18` age = 7.** This is almost certainly a data-entry error and
   needs confirmation from the data owner. Until then, don't use age as a
   covariate for this subject.
3. **D2 age confound.** HC mean age is about 62 (range 55–71) and AD mean age
   is about 80 (range 66–92). Age-related EEG slowing could be classified as
   "AD". The analysis needs an age-controlled check (age-only baseline,
   age-matched subset, or regressing out age), and the paper should discuss it.
4. **Mastoid asymmetry.** M1/M2 are dropped in D1 (30 ch) but kept in D2
   (32 ch), so the forward models differ between datasets. That is
   documented but not fixed, and it matters for any pooled D1+D2 model.
5. **No digitized electrodes.** Coregistration uses the standard 10-20
   template on fsaverage.
6. **D1 unit bug.** The EDF headers say V when the values are µV. This is
   corrected by a ×1e-6 rescale.
7. **Small N.** There are 43 subjects in total (20 AD / 23 HC), so
   subject-level CV with confidence intervals is mandatory, and any
   window-level accuracy will be inflated.
8. **Per-subject quality notes** (bad channels, eye blinks, movement) are in
   the `Comments` column. Decide whether bad channels get interpolated before
   source modeling. Currently they are not handled explicitly.
