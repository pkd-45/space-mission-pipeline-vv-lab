from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AstrometrySeries:
    time_days: np.ndarray
    measured_uas: np.ndarray
    uncertainty_uas: np.ndarray
    nuisance_matrix: np.ndarray
    injected_amplitude_uas: float
    period_days: float


@dataclass(frozen=True)
class AstrometryFit:
    amplitude_uas: float
    amplitude_sigma_uas: float
    delta_bic: float
    residual_rms_uas: float
    detected: bool


def simulate_astrometry_series(
    seed: int,
    amplitude_uas: float = 1.0,
    n_epochs: int = 180,
    period_days: float = 365.25,
) -> AstrometrySeries:
    """Create a transparent differential-astrometry time series.

    Values are illustrative microarcsecond-scale *analysis units*. This is not a
    TOLIMAN optical or spacecraft simulator. The purpose is to exercise nuisance
    regression, injection-recovery, model comparison, and V&V reporting.
    """
    rng = np.random.default_rng(seed)
    time_days = np.sort(rng.uniform(0.0, 3.2 * 365.25, n_epochs))
    phase = rng.uniform(0, 2 * np.pi)

    thermal = np.sin(2 * np.pi * time_days / 97.0 + rng.uniform(0, 2 * np.pi))
    pointing = rng.normal(0, 1, n_epochs)
    pointing = np.convolve(pointing, np.ones(7) / 7, mode="same")
    psf_width = 0.6 * np.sin(2 * np.pi * time_days / 43.0) + rng.normal(0, 0.35, n_epochs)
    nuisance = np.column_stack([thermal, pointing, psf_width])

    # Slowly varying binary/orbital baseline over the compact demonstration span.
    centered = (time_days - np.mean(time_days)) / np.ptp(time_days)
    baseline = 4.0 * centered + 1.2 * centered**2
    planet = amplitude_uas * np.sin(2 * np.pi * time_days / period_days + phase)
    systematic = nuisance @ np.array([0.55, 0.8, -0.45])
    sigma = 1.05 + 0.25 * rng.random(n_epochs)
    measured = baseline + planet + systematic + rng.normal(0, sigma)

    return AstrometrySeries(
        time_days=time_days,
        measured_uas=measured.astype(np.float64),
        uncertainty_uas=sigma.astype(np.float64),
        nuisance_matrix=nuisance.astype(np.float64),
        injected_amplitude_uas=float(amplitude_uas),
        period_days=float(period_days),
    )


def _weighted_fit(y: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weight = 1.0 / np.square(sigma)
    xtw = x.T * weight
    fisher = xtw @ x
    covariance = np.linalg.pinv(fisher)
    beta = covariance @ (xtw @ y)
    residual = y - x @ beta
    chi2 = float(np.sum(np.square(residual / sigma)))
    return beta, covariance, chi2


def fit_astrometric_signal(series: AstrometrySeries) -> AstrometryFit:
    time = series.time_days
    centered = (time - np.mean(time)) / np.ptp(time)
    base = np.column_stack(
        [
            np.ones_like(time),
            centered,
            centered**2,
            series.nuisance_matrix,
        ]
    )
    omega = 2 * np.pi / series.period_days
    signal_cols = np.column_stack([np.sin(omega * time), np.cos(omega * time)])
    full = np.column_stack([base, signal_cols])

    _, _, chi2_null = _weighted_fit(series.measured_uas, series.uncertainty_uas, base)
    beta, covariance, chi2_full = _weighted_fit(series.measured_uas, series.uncertainty_uas, full)
    sin_coeff, cos_coeff = beta[-2:]
    amplitude = float(np.hypot(sin_coeff, cos_coeff))

    grad = np.array([sin_coeff, cos_coeff], dtype=float) / max(amplitude, 1e-12)
    signal_cov = covariance[-2:, -2:]
    amplitude_sigma = float(np.sqrt(max(grad @ signal_cov @ grad, 0.0)))

    n = len(time)
    bic_null = chi2_null + base.shape[1] * np.log(n)
    bic_full = chi2_full + full.shape[1] * np.log(n)
    delta_bic = float(bic_null - bic_full)
    residual = series.measured_uas - full @ beta
    rms = float(np.sqrt(np.mean(np.square(residual))))
    detected = bool(delta_bic >= 10.0 and amplitude >= 3.0 * max(amplitude_sigma, 1e-12))
    return AstrometryFit(amplitude, amplitude_sigma, delta_bic, rms, detected)


def injection_recovery(
    amplitudes_uas: np.ndarray,
    n_trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    recovered = []
    median_bias = []
    median_delta_bic = []
    for index, amplitude in enumerate(amplitudes_uas):
        fits = [
            fit_astrometric_signal(
                simulate_astrometry_series(seed + index * 10000 + trial, float(amplitude))
            )
            for trial in range(n_trials)
        ]
        recovered.append(np.mean([fit.detected for fit in fits]))
        median_bias.append(np.median([fit.amplitude_uas - amplitude for fit in fits]))
        median_delta_bic.append(np.median([fit.delta_bic for fit in fits]))
    return {
        "amplitude_uas": np.asarray(amplitudes_uas, dtype=float),
        "recovery_fraction": np.asarray(recovered, dtype=float),
        "median_amplitude_bias_uas": np.asarray(median_bias, dtype=float),
        "median_delta_bic": np.asarray(median_delta_bic, dtype=float),
    }
