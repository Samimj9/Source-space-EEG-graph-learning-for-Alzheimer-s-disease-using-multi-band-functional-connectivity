"""Retired from tests/test_reliability.py by the final framework (docs/final/inventory.md): they test D1/D2 or
earlier-suite functionality that is no longer part of the code base. Kept verbatim; not collected."""

def test_experiment_refuses_ephemeral_storage_for_real_runs(tmp_path, monkeypatch):
    real = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    real.manifest = {**real.manifest, "synthetic": False}            # stands in for a real graph run
    with pytest.raises(EphemeralStorageError):
        runner.run_variant(real, lr_variant(1), "within_D1", "/content/experiments")
    monkeypatch.setattr(storage, "in_colab", lambda: True)
    with pytest.raises(EphemeralStorageError):
        runner.run_variant(real, lr_variant(1), "within_D1", tmp_path)
    assert not any(tmp_path.iterdir())
    synth = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    v = lr_variant(1)
    v["hparams"] = {}
    runner.run_variant(synth, v, "within_D1", tmp_path)                # synthetic framework tests are exempt


def test_code_change_is_never_silently_reused_in_experiments(tmp_path, monkeypatch, fit_counter):
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    v = lr_variant(1)
    first = runner.run_variant(store, v, "within_D1", tmp_path)
    real = provenance.source_fingerprint
    monkeypatch.setattr(provenance, "source_fingerprint", lambda parts: "edited" + real(parts)[6:])
    st = runner.experiment_status(store, v, "within_D1", tmp_path)
    assert st["state"] == "incompatible" and st["changed"] == "code"
    with pytest.raises(IncompatibleCheckpoint, match="code"):
        runner.run_variant(store, v, "within_D1", tmp_path)
    fits = fit_counter["fits"]
    again = runner.run_variant(store, v, "within_D1", tmp_path, force=True)    # explicit recomputation
    assert fit_counter["fits"] > fits and (tmp_path / "_superseded").exists()
    assert again["subject_level"]["all"]["roc_auc"]["mean"] == first["subject_level"]["all"]["roc_auc"]["mean"]


