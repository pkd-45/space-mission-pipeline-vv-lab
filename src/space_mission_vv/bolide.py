from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit


CLASS_NAMES = np.asarray(["lightning", "bolide", "glint_or_artifact"])


@dataclass(frozen=True)
class BolideDataset:
    light_curves: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]


def _normalise(curve: np.ndarray) -> np.ndarray:
    baseline = np.percentile(curve, 10)
    shifted = np.maximum(curve - baseline, 0)
    return shifted / max(float(np.max(shifted)), 1e-9)


def _curve_features(curve: np.ndarray, latitude: float, local_hour: float) -> np.ndarray:
    norm = _normalise(curve)
    peak = int(np.argmax(norm))
    half = norm >= 0.5
    duration = float(np.count_nonzero(half))
    rise = float(peak - np.flatnonzero(norm >= 0.2)[0]) if np.any(norm >= 0.2) else 0.0
    tail = float(np.trapezoid(norm[peak:]))
    pre = float(np.trapezoid(norm[: peak + 1]))
    asymmetry = (tail - pre) / max(tail + pre, 1e-6)
    roughness = float(np.mean(np.abs(np.diff(norm))))
    secondary = np.partition(norm, -2)[-2] if len(norm) >= 2 else 0.0
    return np.asarray(
        [
            duration,
            rise,
            asymmetry,
            roughness,
            float(secondary),
            abs(float(latitude)) / 90.0,
            min(abs(local_hour - 12.0), 12.0) / 12.0,
        ],
        dtype=np.float64,
    )


def simulate_bolide_dataset(seed: int, n_groups: int = 900, n_steps: int = 64) -> BolideDataset:
    """Build grouped synthetic GLM-like event snippets.

    The benchmark captures rare-event classification and geospatial/contextual
    features without claiming to reproduce GOES GLM instrument data.
    """
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    time = np.linspace(0, 1, n_steps)

    for group in range(n_groups):
        # Deliberately imbalanced prior, then evaluate with class-balanced metrics.
        label = int(rng.choice([0, 1, 2], p=[0.78, 0.08, 0.14]))
        latitude = float(rng.uniform(-60, 60))
        local_hour = float(rng.uniform(0, 24))
        centre = float(rng.uniform(0.25, 0.75))
        noise = rng.normal(0, 0.025, n_steps)

        if label == 0:  # lightning: multi-pulse, jagged, often longer
            curve = np.zeros(n_steps)
            n_pulses = int(rng.integers(2, 6))
            for _ in range(n_pulses):
                c = centre + rng.normal(0, 0.08)
                width = rng.uniform(0.012, 0.045)
                curve += rng.uniform(0.35, 1.0) * np.exp(-0.5 * ((time - c) / width) ** 2)
            curve += 0.08 * np.sin(2 * np.pi * rng.uniform(8, 18) * time + rng.uniform(0, 2 * np.pi))
        elif label == 1:  # bolide: sharp rise and smoother exponential-like decay
            width = rng.uniform(0.018, 0.045)
            rise = np.exp(-0.5 * ((time - centre) / width) ** 2)
            tail = np.where(time >= centre, np.exp(-(time - centre) / rng.uniform(0.05, 0.14)), 0.0)
            curve = rng.uniform(0.75, 1.2) * (0.65 * rise + 0.55 * tail)
            if rng.random() < 0.35:
                curve += rng.uniform(0.12, 0.35) * np.exp(-0.5 * ((time - centre - 0.06) / 0.025) ** 2)
        else:  # glint/artifact: symmetric, narrow, occasionally clipped or repeated
            width = rng.uniform(0.008, 0.025)
            curve = rng.uniform(0.65, 1.15) * np.exp(-0.5 * ((time - centre) / width) ** 2)
            if rng.random() < 0.45:
                curve += rng.uniform(0.25, 0.65) * np.exp(-0.5 * ((time - centre - 0.12) / width) ** 2)

        curve = np.maximum(curve + noise, 0.0)
        curves.append(curve.astype(np.float32))
        features.append(_curve_features(curve, latitude, local_hour))
        labels.append(label)
        groups.append(group // 3)  # related operational batches stay together

    return BolideDataset(
        light_curves=np.stack(curves),
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        feature_names=(
            "duration_above_half_peak",
            "rise_time",
            "temporal_asymmetry",
            "roughness",
            "second_peak_fraction",
            "absolute_latitude_scaled",
            "distance_from_local_noon_scaled",
        ),
    )


def fit_and_evaluate(dataset: BolideDataset, seed: int) -> dict[str, object]:
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_index, holdout_index = next(first.split(dataset.features, dataset.labels, dataset.groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_relative, test_relative = next(
        second.split(
            dataset.features[holdout_index],
            dataset.labels[holdout_index],
            dataset.groups[holdout_index],
        )
    )
    val_index = holdout_index[val_relative]
    test_index = holdout_index[test_relative]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            C=1.2,
            random_state=seed,
        ),
    )
    model.fit(dataset.features[train_index], dataset.labels[train_index])
    probability = model.predict_proba(dataset.features[test_index])
    prediction = np.argmax(probability, axis=1)
    # Mission-style review threshold: require stronger evidence before assigning
    # the rare bolide class, then choose between common nuisance classes.
    bolide_threshold = 0.65
    prediction[probability[:, 1] >= bolide_threshold] = 1
    below = probability[:, 1] < bolide_threshold
    prediction[below] = np.where(probability[below, 0] >= probability[below, 2], 0, 2)
    truth = dataset.labels[test_index]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        prediction,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "bolide_precision": float(precision[1]),
        "bolide_recall": float(recall[1]),
        "bolide_f1": float(f1[1]),
        "bolide_brier": float(brier_score_loss((truth == 1).astype(int), probability[:, 1])),
        "bolide_review_threshold": bolide_threshold,
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(len(CLASS_NAMES))).tolist(),
        "n_train": int(len(train_index)),
        "n_validation": int(len(val_index)),
        "n_test": int(len(test_index)),
        "group_overlap_train_test": int(len(set(dataset.groups[train_index]) & set(dataset.groups[test_index]))),
    }
    ranking = np.argsort(probability[:, 1])[::-1]
    metrics["top_candidates"] = [
        {
            "test_row": int(test_index[index]),
            "truth": str(CLASS_NAMES[truth[index]]),
            "predicted": str(CLASS_NAMES[prediction[index]]),
            "bolide_probability": float(probability[index, 1]),
        }
        for index in ranking[:12]
    ]
    metrics["model"] = model
    metrics["test_index"] = test_index
    metrics["prediction"] = prediction
    metrics["probability"] = probability
    return metrics
