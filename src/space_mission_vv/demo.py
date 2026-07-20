from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .astrometry import fit_astrometric_signal, injection_recovery, simulate_astrometry_series
from .bolide import CLASS_NAMES, fit_and_evaluate, simulate_bolide_dataset
from .provenance import write_manifest


def run_demo(outdir: str | Path, seed: int, n_astrometry_trials: int, n_bolide_groups: int) -> dict[str, object]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    example = simulate_astrometry_series(seed, amplitude_uas=1.0)
    example_fit = fit_astrometric_signal(example)
    amplitudes = np.asarray([0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6])
    recovery = injection_recovery(amplitudes, n_astrometry_trials, seed + 1000)

    bolide_data = simulate_bolide_dataset(seed + 2000, n_groups=n_bolide_groups)
    bolide = fit_and_evaluate(bolide_data, seed + 3000)

    metrics = {
        "scope": "controlled synthetic software-validation benchmark",
        "astrometry": {
            "example_injected_amplitude_uas": example.injected_amplitude_uas,
            "example_recovered_amplitude_uas": example_fit.amplitude_uas,
            "example_amplitude_sigma_uas": example_fit.amplitude_sigma_uas,
            "example_delta_bic": example_fit.delta_bic,
            "example_residual_rms_uas": example_fit.residual_rms_uas,
            "recovery": {key: value.tolist() for key, value in recovery.items()},
        },
        "bolide": {key: value for key, value in bolide.items() if key not in {"model", "test_index", "prediction", "probability"}},
        "v_and_v_gates": {
            "astrometry_zero_signal_false_detection_le_0_10": bool(recovery["recovery_fraction"][0] <= 0.10),
            "astrometry_high_signal_recovery_ge_0_85": bool(recovery["recovery_fraction"][-1] >= 0.85),
            "bolide_balanced_accuracy_ge_0_75": bool(bolide["balanced_accuracy"] >= 0.75),
            "bolide_precision_ge_0_65": bool(bolide["bolide_precision"] >= 0.65),
            "no_group_leakage": bool(bolide["group_overlap_train_test"] == 0),
        },
    }
    metrics["v_and_v_pass"] = bool(all(metrics["v_and_v_gates"].values()))

    metrics_path = out / "reference_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with (out / "astrometry_recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(recovery.keys())
        writer.writerows(zip(*recovery.values(), strict=True))

    with (out / "bolide_candidate_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bolide["top_candidates"][0].keys()))
        writer.writeheader()
        writer.writerows(bolide["top_candidates"])

    _make_figure(example, example_fit, recovery, bolide_data, bolide, metrics, out)
    write_manifest(
        out,
        seed,
        [
            "reference_metrics.json",
            "astrometry_recovery.csv",
            "bolide_candidate_ranking.csv",
            "space_mission_pipeline_vv_summary.png",
        ],
    )
    return metrics


def _make_figure(example, fit, recovery, bolide_data, bolide, metrics, out: Path) -> None:
    fig = plt.figure(figsize=(13.5, 8.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])

    ax_series = fig.add_subplot(grid[0, 0])
    ax_series.errorbar(
        example.time_days,
        example.measured_uas,
        yerr=example.uncertainty_uas,
        fmt=".",
        markersize=3,
        alpha=0.55,
        linewidth=0.5,
    )
    ax_series.set_title(
        f"Precision-astrometry injection\nAinj=1.00, Arec={fit.amplitude_uas:.2f}+/-{fit.amplitude_sigma_uas:.2f} analysis uas"
    )
    ax_series.set_xlabel("mission day")
    ax_series.set_ylabel("differential displacement [illustrative uas]")

    ax_recovery = fig.add_subplot(grid[1, 0])
    ax_recovery.plot(recovery["amplitude_uas"], recovery["recovery_fraction"], marker="o")
    ax_recovery.axhline(0.9, linestyle="--", linewidth=1)
    ax_recovery.set_ylim(-0.03, 1.03)
    ax_recovery.set_xlabel("injected signal amplitude [illustrative uas]")
    ax_recovery.set_ylabel("detection fraction")
    ax_recovery.set_title("Astrometry injection-recovery V&V")

    ax_curves = fig.add_subplot(grid[:, 1])
    for class_id, name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(bolide_data.labels == class_id)[:4]
        for count, index in enumerate(indices):
            normalised = bolide_data.light_curves[index] / max(float(np.max(bolide_data.light_curves[index])), 1e-6)
            ax_curves.plot(normalised + class_id * 1.35, alpha=0.75, label=name if count == 0 else None)
    ax_curves.set_xlabel("time sample")
    ax_curves.set_yticks([0.35, 1.70, 3.05], CLASS_NAMES)
    ax_curves.set_title("Rare-event light-curve classes\n(synthetic GLM-like snippets)")
    ax_curves.legend(loc="upper right", fontsize=8)

    ax_cm = fig.add_subplot(grid[0, 2])
    cm = np.asarray(bolide["confusion_matrix"])
    image = ax_cm.imshow(cm)
    ax_cm.set_xticks(range(3), ["L", "B", "G"])
    ax_cm.set_yticks(range(3), ["L", "B", "G"])
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_title(
        f"Grouped test confusion matrix\nBA={bolide['balanced_accuracy']:.2f}, bolide F1={bolide['bolide_f1']:.2f}"
    )
    for row in range(3):
        for col in range(3):
            ax_cm.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_gate = fig.add_subplot(grid[1, 2])
    names = [
        "zero-signal false alarm",
        "high-signal recovery",
        "balanced accuracy",
        "bolide precision",
        "group isolation",
    ]
    passed = list(metrics["v_and_v_gates"].values())
    ax_gate.barh(np.arange(len(names)), [1 if flag else 0 for flag in passed])
    ax_gate.set_yticks(np.arange(len(names)), names)
    ax_gate.set_xlim(0, 1.05)
    ax_gate.set_xticks([0, 1], ["FAIL", "PASS"])
    ax_gate.set_title(f"Release V&V gates: {'PASS' if metrics['v_and_v_pass'] else 'FAIL'}")

    fig.suptitle(
        "Space Mission Pipeline V&V Lab: precision astrometry + rare-event satellite classification",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Controlled synthetic benchmark only - not a TOLIMAN optical simulator and not a GOES GLM performance claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(out / "space_mission_pipeline_vv_summary.png", dpi=170)
    plt.close(fig)
