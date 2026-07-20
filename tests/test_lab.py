from pathlib import Path

import numpy as np

from space_mission_vv.astrometry import fit_astrometric_signal, injection_recovery, simulate_astrometry_series
from space_mission_vv.bolide import fit_and_evaluate, simulate_bolide_dataset
from space_mission_vv.demo import run_demo
from space_mission_vv.provenance import sha256_file


def test_astrometry_is_deterministic():
    first = simulate_astrometry_series(123, amplitude_uas=0.8)
    second = simulate_astrometry_series(123, amplitude_uas=0.8)
    assert np.array_equal(first.measured_uas, second.measured_uas)
    assert np.isfinite(first.measured_uas).all()


def test_astrometry_fit_recovers_bright_signal():
    series = simulate_astrometry_series(34, amplitude_uas=1.8)
    fit = fit_astrometric_signal(series)
    assert fit.detected
    assert abs(fit.amplitude_uas - 1.8) < 0.5


def test_injection_recovery_increases():
    result = injection_recovery(np.array([0.0, 0.7, 1.6]), n_trials=40, seed=55)
    assert result["recovery_fraction"][0] <= 0.15
    assert result["recovery_fraction"][-1] >= 0.80
    assert result["recovery_fraction"][-1] > result["recovery_fraction"][1]


def test_bolide_split_has_no_group_leakage():
    dataset = simulate_bolide_dataset(91, n_groups=450)
    metrics = fit_and_evaluate(dataset, 92)
    assert metrics["group_overlap_train_test"] == 0
    assert metrics["balanced_accuracy"] >= 0.70


def test_bolide_dataset_contains_all_classes():
    dataset = simulate_bolide_dataset(93, n_groups=500)
    assert set(np.unique(dataset.labels)) == {0, 1, 2}
    assert dataset.features.shape[1] == 7


def test_full_demo_writes_vv_products(tmp_path: Path):
    metrics = run_demo(tmp_path, seed=2026, n_astrometry_trials=18, n_bolide_groups=360)
    assert metrics["v_and_v_pass"]
    for filename in [
        "reference_metrics.json",
        "astrometry_recovery.csv",
        "bolide_candidate_ranking.csv",
        "space_mission_pipeline_vv_summary.png",
        "product_manifest.json",
    ]:
        assert (tmp_path / filename).exists()
        assert (tmp_path / filename).stat().st_size > 0


def test_checksum_changes_after_mutation(tmp_path: Path):
    path = tmp_path / "product.txt"
    path.write_text("first", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("second", encoding="utf-8")
    second = sha256_file(path)
    assert first != second
