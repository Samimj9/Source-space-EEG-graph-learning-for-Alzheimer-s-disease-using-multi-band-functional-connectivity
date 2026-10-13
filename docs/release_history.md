# Code versions before the first real-data run

| Version | Commit | Status |
|---|---|---|
| `pre-real-data-v1` (annotated tag, created 2026-10-12) | `0589369a3d3100380f3adcf7218198446f7c6621` | **Historical baseline.** Frozen methodology, graph pipeline, experiments and visualization framework with checkpointing and reliability safeguards. Kept unchanged in the branch history. |
| Final pre-real-data version | the commit that first appears in `<outputs>/CODE_COMMIT.txt` on Drive (also in every graph and experiment manifest as `git_commit`) | **Analysis version.** Adds the per-experiment Colab notebooks, the pre-real-data audit fixes (entry 33), bounded memory (34), progress bars (35), the connectivity band-pass transition widths (36) and the index-based EDF channel mapping through the `.loc` layout (37). |

Note on the tag: the `pre-real-data-v1` tag exists in the author's local clone, but pushing tags is
not possible from the coding session. To publish it, run from any clone:

```bash
git tag -a pre-real-data-v1 0589369a3d3100380f3adcf7218198446f7c6621 \
  -m "Frozen methodology, graph pipeline, experiments and visualization framework before first real-data execution"
git push origin pre-real-data-v1
```

## How the analysis version is fixed in Colab

1. The first notebook session clones the branch and records its commit in
   `<outputs>/CODE_COMMIT.txt`.
2. Every later session, in any notebook, checks out that commit, so all graphs, experiments
   and figures come from one code version (CLAUDE.md rule 7).
3. If a pin exists but no graph or experiment checkpoint has been completed under it (for
   example, an earlier setup-only session), it protects nothing. It is moved automatically to
   the current commit and reported. The first real run therefore always records the newest
   commit, never an outdated one.
4. Once results exist, the pin moves only with `UPDATE_CODE = True` in the first cell. That
   deliberately recomputes unfinished units.
