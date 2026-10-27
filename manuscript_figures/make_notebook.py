"""Writes D3_Manuscript_Figure_Generation.ipynb (the notebook only calls build_all_figures.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("# D3 manuscript figures (reads saved outputs only)\nNo model is trained or evaluated here. Every plotted number comes from a saved file; AUC, DeLong CI and confusion counts recomputed from saved participant predictions are cross-checked against the saved results table, and the run aborts on a mismatch. Exp0 / Exp1 are excluded from every figure. Output goes to a separate folder; the code refuses to write inside `FINAL_v1`.\n\nRun the cells top to bottom (Runtime > Run all).")
code("""# 1. Mount Drive and locate the project (read-only use)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROCESSED = Path('/content/drive/MyDrive/AD-LEB/processed')
assert (PROCESSED / 'FINAL_v1').exists(), 'FINAL_v1 not found under AD-LEB/processed'
print('FINAL_v1 found:', PROCESSED / 'FINAL_v1')""")
code("""# 2. Get the figure package and its requirements
import subprocess, sys, os
REPO = Path('/content/d3_repo')
if not REPO.exists():
    subprocess.run(['git','clone','--depth','1','--branch','claude/a3-final-analysis',
                    'https://github.com/Samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity.git', str(REPO)], check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(REPO/'manuscript_figures'/'requirements.txt')], check=True)
sys.path.insert(0, str(REPO/'manuscript_figures')); os.chdir(REPO/'manuscript_figures')""")
code("""# 3. Locate the saved inputs and validate them (fails clearly if something is missing)
import build_all_figures as B
cfg = B.Config(out=Path('/content/manuscript_figures_out/generated'), processed=PROCESSED)
hit = next(iter(PROCESSED.rglob('fig1_primary_forest.png')), None)
cfg.existing_fig_dir = hit.parent if hit else None
print('existing figure folder:', cfg.existing_fig_dir)
d = B.load_results(cfg)                       # development_results_all.csv, required columns checked
info = B.validate_inventory(d)                # 52 participants for every GNN row
print(info['n_all'], 'results;', info['n_gnn'], 'GNN results;', info['n_classical_excluded'], 'classical rows excluded')
for u in (B.PRIMARY, B.EXP5_R18, B.EXP4_REF, B.HOLDOUT_UNIT):
    r = B.pick(d, u); print(u.key, r.experiment, r.variant, r.inverse, r['mode'], r.atlas, r.graph, round(r.auc, 6), r.rel_dir)""")
code("""# 4. Check the saved participant predictions (one out-of-fold score per participant) against the saved table
for u in (B.PRIMARY, B.EXP5_R18, B.EXP4_REF):
    p = B.load_predictions(cfg, d, u)          # raises if AUC / confusion differ from the table
    print(u.key, len(p), 'participants; recomputed AUC matches the saved table')""")
code("""# 5. Build every figure (missing inputs are reported, never replaced)
manifest = B.build(cfg)                        # figure_manifest.csv, source_inventory.csv, run_info.json written next to generated/
import pandas as pd
display(pd.DataFrame(manifest))""")
code("""# 6. Optional: cortical-surface rendering of an atlas (needs mne + the fsaverage files cached by notebook 02)
# out = B.render_cortical_surface(cfg, 'lobes12'); print(out)""")
code("""# 7. Preview the outputs
from IPython.display import Image, display
for f in sorted(cfg.out.glob('*.png')):
    print(f.name); display(Image(str(f), width=900))""")
nb.cells = C
nbf.validate(nb)
nbf.write(nb, "D3_Manuscript_Figure_Generation.ipynb")
print("notebook written and valid")
