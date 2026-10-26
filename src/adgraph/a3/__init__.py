"""Amendment A3 (methodology log entry 61): final-analysis integration, frozen-model evaluation, consolidated reporting.

A3 lives OUTSIDE the packages that define an experiment unit's code identity (``checkpoint.CODE_SCOPES["experiment"]``:
config.py, experiments, models, evaluation, graphs, features, qc), so adding it changes neither the source fingerprint of
the A2 pin nor of the release pin (tested): no completed Exp 0-8a checkpoint can become "incompatible" because of A3.
A3 never trains in the development setting, never writes into the release or A2 results, and never moves a pin. Its own
outputs live in ``<processed>/FINAL_v1/amendments/A3_final/``; the only files it writes elsewhere are the two frozen
records the freeze creates (once, checksummed), in the format each pinned code version already reads.
"""
