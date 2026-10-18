# Archived code of the earlier work (not part of the final framework)

Moved here with `git mv` when the final framework was assembled (methodology log entry 58; `docs/final/inventory.md`).
Nothing here is importable or collected by pytest. To run any of it, check out the commit that produced the result
(e.g. tag `backup/pre-unification-607b5ea` or branch `claude/ecstatic-davinci-a2e8mh`), where every file is at its
original place.

| Folder | Content |
|---|---|
| `src/adgraph/` | D1 / D2 figures (`visualization`), EdgeWeightedGAT, WeightedGraphSAGE, age covariates, cross-dataset gate, dataset-shift diagnostic, experiment comparison and cost estimation |
| `configs/` | earlier experiment configs (D1 / D2, D3, LX, WX, ES200, R1), figure inventory, run plan, the LX / WX store configurations (reproduced exactly by `configs/source_graphs.yaml`) |
| `scripts/` | per-suite runners, report scripts, pin tools and the earlier notebook generator |
| `notebooks/` | every earlier notebook |
| `tests/` | tests of functionality that no longer exists, verbatim |

The legacy D1 / D2 graph builder itself is NOT archived: it stays importable and tested as `src/adgraph/legacy/`.
