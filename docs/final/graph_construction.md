# Graph construction before GNN classification

Code `adgraph.graphs.construction`, `adgraph.graphs.diagnostics`, `adgraph.experiments.selection`; configs
`configs/final/graphs.yaml`, `configs/final/selection.yaml`; notebook `03_graph_construction`. Methodology log entry 58.

## 1. Input

Per window and band, the stores hold weighted 68 x 68 PLV and |ciPLV| matrices (symmetric, zero diagonal, in [0, 1];
the signed ciPLV is never an adjacency). The dense weighted graph is the reference. Every construction operates per
window and per band graph, keeps the store's band order and ROI order, and returns the same shape.

## 2. Constructions (pre-specified)

| name | definition |
|---|---|
| `dense` | the weighted connectivity as is (reference, default) |
| `knn_union_f10/f20/f30` | each node keeps its k strongest edges; i~j if either keeps the edge (symmetric, degree >= k) |
| `knn_mutual_f10/f20/f30` | i~j only if both keep it (degree <= k); a node left without an edge gets its strongest edge back, symmetrically (`min_degree: 1`) |
| `proportional_f20` | the strongest round(d R (R - 1) / 2) edges of each graph, density d = k / (R - 1) |
| `contiguity` | functional weights on pairs of cortically contiguous regions only (fsaverage mesh adjacency; `source/atlas_tables.json`) |
| `contiguity_knn_union_f20` | union kNN restricted to the contiguity mask |
| `feature_similarity_knn_union_f20` | a DIFFERENT graph: per window, Gaussian similarity of the ROIs' z-scored [log PSD, relative PSD, DE] vectors of all bands, then union kNN; the same graph for every band; never a substitute for connectivity |

**k is relative to the number of ROIs:** k = max(1, round(f (R - 1))), f in {0.10, 0.20, 0.30}: k = 7 / 13 / 20 for
68 ROIs, 3 / 7 / 10 for 34, 2 / 3 / 5 for 18 (tested). Small, sparse-to-moderate densities (about 10-45 % of the possible
edges for union kNN) on either side of the commonly used 10-30 % proportional thresholds, fixed before any result.

**Guarantees (tested, `tests/test_construction.py`):** symmetric (except the analysis-only `symmetrize: none`), zero
diagonal (self-loops are added by the GCN normalization D^-1/2 (A + I) D^-1/2, unchanged), retained edges keep their
exact weights (never binarized), no negative weights, zero-weight edges never kept, no isolated node for kNN
constructions, ties broken deterministically by ROI index, ROI relabelling commutes, band graphs independent, the only
fitted construction (`quantile`, not in the comparison) fitted on training windows only. Node strength is always the
mean functional connectivity of the dense band graph, the same for every construction, so a construction changes
message passing only. Atlas-relative k and contiguity work for every atlas of `source.atlas`.

## 3. Compatibility

Every GNN architecture takes the constructed adjacency (MultiBandGNN GCN encoders, EARM-GNN as the prior of its edge
gates, Xu dual GCN layer 1); PSD / DE node features are unchanged; LCMV and wMNE stores are treated identically.

## 4. Structural diagnostics (label-blind, development participants only)

`python scripts/final.py graphs --inverse <lcmv|wmne> --mode B` writes, per construction, measure and band, on up to
10 windows of each of the 52 development participants: density, mean / SD / min / max degree, isolated nodes, graphs
with an isolated node, connected components (mean, max) and fraction connected, retained fraction of the dense
graph's total weight, symmetry / diagonal / minimum weight, edges per graph relative to dense, the node-feature
preservation below, and the edge-set
stability (mean Jaccard of retained edges within a participant vs between participants). The holdout is never read;
the diagnosis is never read (tested). **Computational requirements:** the GNNs use dense batched matrix products, so
their cost does not fall with sparsity (it is the same R x R product); sparsity changes the information propagated,
not the run time. The diagnostics table reports the edge counts for completeness.

**Node-feature preservation** (`diagnostics.propagation`, columns `gcn_self_weight`,
`feature_variance_retained_1/2`): the question for PSD / DE node features. A GCN layer replaces each ROI's feature by
A_hat x with A_hat = D^-1/2 (A + I) D^-1/2, whose diagonal is 1 / (1 + d_i). With dense PLV weights (store means
0.27 / 0.31 over 67 neighbours, d_i ≈ 18-21) an ROI keeps about 5 % of its own feature per layer and the rest is a
connectivity-weighted average of all other ROIs: the ROI-specific spectrum, which is where PSD / DE differ between
ROIs, is largely averaged away (over-smoothing). Measured on PLV-like synthetic weights with the store means:

| construction | GCN self-weight | across-ROI variance kept after 1 step (independent features; lower bound) |
|---|---|---|
| dense | 0.046-0.052 | ≈ 0.003 |
| union kNN f10 (k = 7) | 0.19-0.21 | ≈ 0.10 |
| union kNN f20 (k = 13) | 0.13-0.14 | |
| union kNN f30 (k = 20) | 0.09-0.10 | |
| mutual kNN f20 (k = 13) | 0.16-0.17 | |

Real PSD / DE are spatially smooth (neighbouring ROIs share leakage), so the real retained variance is higher than
this lower bound; notebook 03 reports it on the stores. **This is a mechanism, not evidence that kNN classifies
better:** a model that uses the graph mainly as a smoother may do well with dense graphs, and the Exp 6 / 8 models
also carry node features through non-propagated paths. Hence the empirical development-only comparison of §5.

**Normalization verified** (`tests/test_construction.py`): the GCN operator is exactly D^-1/2 (A + I) D^-1/2,
symmetric, finite, with exactly one unit self-loop added; an isolated node keeps only itself (A_hat_ii = 1, no NaN).
EARM-GNN normalizes its gated graph D^-1/2 (A ⊙ G) D^-1/2 without self-loops (its residual blocks carry the node's
own state). Constructions never rescale retained weights; normalization belongs to the model.

**Directed top-k vs symmetric kNN:** a per-node top-k graph is not symmetric (i may keep j while j does not keep
i), which breaks the symmetric normalization and the undirected interpretation of PLV / |ciPLV|; it exists only for
analysis (`symmetrize: none`, reported `symmetric: false`) and is not a candidate. Union kNN (degree >= k, every node
keeps its own k strongest edges) and mutual kNN (degree <= k, only reciprocal edges, isolated nodes repaired) are the
two symmetric versions; weighted proportional thresholding keeps the strongest edges globally and may isolate weakly
connected ROIs (reported by the diagnostics; kept as the conventional alternative).

## 5. Development-only choice

A construction other than `dense` is adopted only by the pre-specified rule of `configs/final/selection.yaml`, run
ONCE, on the atlas the author recorded after the ROI comparison (notebook 04 before 03; `atlas: selected`; k scales
with that atlas's ROI count). It is an exploratory, pre-specified decision rule, not a test of statistical
superiority; the per-cell AUCs carry their DeLong / bootstrap CIs in the result tables.

* reference models: Exp 4 `plv__psd` (band-averaged graph) and Exp 5 `plv__separate__mean` (one graph per band),
  both inverses, mode B, development CV only (`--settings xu_dev_cv`); 4 cells per construction;
* the best eligible candidate replaces `dense` only if its mean participant AUC over the 4 cells is >= dense + 0.02
  AND it is better than dense in >= 3 of the 4 cells; otherwise `dense` is kept;
* every candidate must have completed every cell (an incomplete comparison keeps `dense`);
* the feature-similarity graph is reported in the same table but is not eligible;
* the decision and every number it used are written once to `FINAL_v1/decisions/graph_construction.json` and never
  overwritten; Exp 2-8 then run with `--graph selected`.

kNN is therefore not declared better than dense by default: a sparser graph must earn its place on development data,
by a margin, consistently across models and inverses. The holdout plays no part (tested).

**Cost of the comparison** (notebook 03, mode B, Colab CPU estimates from measured fit times, §FINAL_RUN_GUIDE):
11 constructions x 2 models x 2 inverses x 5 folds = 220 fold fits, about 3 min (Exp 4) and 10 min (Exp 5) each on
CPU, i.e. about 24 CPU hours; GPU faster (not measured).
