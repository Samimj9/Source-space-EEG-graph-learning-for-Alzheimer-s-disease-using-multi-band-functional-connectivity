# Methods draft (journal manuscript)

Working draft of the manuscript Methods. Every statement describes committed code and configuration;
numbers come from committed configs or from saved, validated outputs. Sections still to be drafted are
marked TODO. The boundary-aware signal-processing subsection is in `docs/boundary_processing.md` §10
(with its Supplementary Methods S1–S8).

Code commits: graph dataset `e637368` (graph run `graphs_v4_33faa4b89a5d`, `graph_run_id
6d75de9975f0d9ae`); classical baselines Exp 0/1 `e637368`; fixed-hyperparameter GNN experiments: the
commit recorded in their manifests (see the completion report).

---

## 2.x Participants and recordings — TODO

## 2.x Preprocessing and boundary-aware signal processing

See `docs/boundary_processing.md` §10 ("Boundary-Aware Signal Processing and Functional Connectivity
Estimation"), which reports the frozen dataset: 43 recordings, 1,527 accepted 20-s windows (dataset 1:
907; dataset 2: 620; AD: 729; HC: 798).

## 2.x Source reconstruction, connectivity and graph construction — TODO

## 2.x Classification models

**Spectral baselines.** Experiment 0 used electrode-level band power (30 electrodes × 4 bands × [log
absolute, relative] PSD) and Experiment 1 regional source-space band power (68 regions × 4 bands × [log
absolute, relative]). Each was classified by L2-regularized logistic regression and by a linear support
vector machine. Features were standardized with statistics of the training windows of each fold. The
regularization parameter C ∈ {0.001, 0.01, 0.1, 1, 10} was selected by subject-level inner 4-fold cross-
validation within each outer training set (selection criterion: subject-level ROC AUC). The SVM score was
its signed decision-function margin (threshold 0); logistic regression used P(AD) (threshold 0.5).

**Graph neural networks.** Each window was represented by one graph per frequency band with the 68
regions as nodes and the band-specific PLV or |ciPLV| matrix as weighted adjacency; PLV and |ciPLV| were
analysed separately and never combined. The experiments differed in the graphs and node features:

| Experiment | Graph(s) | Node features (per region) |
|---|---|---|
| 2 | one band (θ, α1, α2 or β), each a separate model | constant |
| 3 | one band, each a separate model | log and relative PSD of the same band |
| 4 | mean of the four band adjacencies | log and relative PSD of all four bands |
| 5 | four band graphs, shared encoder, mean or attention fusion | log and relative PSD of the graph's band |
| 6 | four band graphs, shared encoder, attention fusion | log PSD, relative PSD and node strength of the graph's band |

The graph encoder was a two-layer graph convolutional network (GCN) on the symmetrically normalized
weighted adjacency with self-loops, D^{−1/2}(A + I)D^{−1/2}, with 32 hidden units and dropout 0.3,
followed by mean and max pooling over regions. In multi-band models a single encoder was shared across
bands and the band embeddings were fused by their mean or by learned attention weights. A two-layer
perceptron with 32 hidden units produced the AD logit. Node features were z-scored per band graph, region
and feature with statistics of the training windows of each fold.

**Fixed hyperparameters.** All graph neural networks were trained with one fixed, pre-specified
configuration, without any hyperparameter search: Adam optimizer (learning rate 10⁻³, β = (0.9, 0.999),
weight decay 10⁻⁴), batch size 32, 60 epochs, no early stopping; the model after the final epoch was
evaluated. The loss was binary cross-entropy weighted so that every training subject and both classes
contributed equally regardless of the number of windows. These values were the defaults specified before
any graph-model result was available. *(Secondary analyses, Tier 2: EdgeWeightedGAT, a graph attention
network with edge weights in the attention logits and messages, 4 heads × 8 units; WeightedGraphSAGE
with weighted sum aggregation; same fixed training configuration.)*

**Ablations of the full model (Experiment 6).** Removing PSD features, removing node strength, replacing
attention by mean fusion, and removing each band in turn (PLV; within-dataset and cross-dataset
settings).

## 2.x Evaluation

**Subject-level cross-validation.** All splitting was by subject; all windows of a subject were always in
the same partition. Within each dataset (primary settings S1, dataset 1, and S2, dataset 2) and in the
pooled cohort (secondary setting S3, stratified by dataset × diagnosis), models were evaluated by
repeated stratified 5-fold cross-validation over subjects. The spectral baselines used 20 repeats
(seeds 0–19); the graph neural networks used 10 repeats (seeds 0–9), whose subject partitions are
identical to the first 10 repeats of the baselines. In each outer fold every data-dependent transform
(feature standardization, node-feature scaling, hyperparameter selection for the baselines) was fitted
on the training subjects only, and each graph neural network was trained exactly once on the training
subjects of the fold. Cross-dataset generalization (S4) was assessed by training on all subjects of one
dataset and testing once on all subjects of the other (both directions), after the pre-specified
pipeline-identity checks were passed.

**Metrics.** Window scores were averaged per subject. Subject-level ROC AUC and balanced accuracy were
the primary metrics; sensitivity (AD), specificity (HC), F1 (AD) and the Matthews correlation coefficient
were secondary. Within each repeat, out-of-fold subject predictions from the five folds were pooled and
the metric computed once; results are reported as mean ± SD over repeats. 95 % confidence intervals were
obtained by 2,000 bootstrap resamples of subjects of the subject scores averaged over repeats. Decision
thresholds were fixed a priori (0.5 for probabilities, 0 for SVM margins) and never tuned. Window-level
results are secondary. Permutation tests were not performed; no significance against chance is claimed.

**Protocol amendment.** The evaluation design was specified before classification (version 1.0) and
amended twice before the results reported here: the boundary-aware graph construction (v1.4) and,
for computational feasibility, the graph-model training protocol (v1.5). The original design selected
graph-model hyperparameters (hidden size 16 or 32; weight decay 10⁻⁴ or 10⁻³) by nested subject-level
4-fold cross-validation with 20 repeats. Because this required 17 network fits per outer fold (several
weeks of GPU time for all experiments), it was replaced by the fixed configuration above, with 10
repeats and a reduced pre-specified set of model variants. The amendment was made after the spectral
baselines and one nested cross-validation graph-model result (single-band PLV, theta, GCN, dataset 1)
had been observed; that result is not reported, and no model, band, fusion method or experiment was
selected on the basis of held-out performance. The final graph-model evaluation therefore uses fixed
hyperparameters without inner hyperparameter optimization.

## 2.x Software and reproducibility — TODO
