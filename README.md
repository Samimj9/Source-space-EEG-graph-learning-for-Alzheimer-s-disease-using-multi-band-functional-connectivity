# Source-space EEG graph learning for Alzheimer's disease

Code for a research paper on classifying Alzheimer's disease from resting-state
EEG using source-space, multi-band functional-connectivity graphs and graph
neural networks.

**Status:** graph construction (v3) and the experiment framework (Exp 0–6) are implemented and tested on synthetic data only. Nothing has been run on the real datasets yet.

- Project rules and layout: [`CLAUDE.md`](CLAUDE.md)
- Datasets and known issues: [`docs/data_inventory.md`](docs/data_inventory.md)
- Evaluation design (frozen): [`DESIGN.md`](DESIGN.md)
- Experiments, tensor shapes, leakage rules: [`docs/experiments.md`](docs/experiments.md)
- Methodological choices: [`docs/methods_decisions.md`](docs/methods_decisions.md)

Data are not distributed with this repository. The pipeline runs in Google
Colab against a mounted Google Drive. See `configs/paths.example.yaml`.
