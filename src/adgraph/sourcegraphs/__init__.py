"""Source-space graph stores of ds004504 for both inverse solutions (LCMV and wMNE), one code path.

Formerly ``adgraph.lcmvxu`` (the LCMV-Xu suite, methodology log entry 50, and its matched wMNE counterpart, entry 55).
The inverse is selected by the store configuration (``configs/source_graphs.yaml``, ``source.method``); everything
else (recordings, segments, selection, windows, rejection, bands, PSD / DE / PLV / |ciPLV|, ROIs, split) is shared.
Modules: ``registry`` (participants, sensor geometry, store discovery), ``recording`` (segments, selection),
``lcmv`` / ``wmne`` (ROI operators), ``features`` (per-window features), ``build`` (resumable store build),
``correct`` (the historical corr1 correction of the LCMV stores), ``validate``, ``match``, ``protocol``.
"""
