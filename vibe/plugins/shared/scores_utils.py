"""Shared utilities for normalizing and processing scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vibe.results import ScoreEntry


def normalize_scalar(score: float, score_min: float, score_max: float) -> float:
    """Normalize a scalar score to a strict [0, 1] range."""
    if score_max <= score_min:
        return 0.0
    return float(np.clip((score - score_min) / (score_max - score_min), 0.0, 1.0))


def load_samples_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an npz samples file for percentile interpolation."""
    with np.load(path, allow_pickle=False) as data:
        arr = np.asarray(data["arr_0"], dtype=np.float32)
        x, y = np.asarray(arr[0], dtype=np.float32), np.asarray(arr[1], dtype=np.float32)

    order = np.argsort(x)
    x, y = x[order], y[order]
    min_x = min(0.0, float(x[0]) - 1e-6)
    x = np.concatenate(([min_x], x, [x[-1] + 1e-6])).astype(np.float32, copy=False)
    y = np.concatenate(([0.0], y, [1.0])).astype(np.float32, copy=False)
    return x, y


def interp_percentile(value: float, x: np.ndarray, y: np.ndarray) -> float:
    """Interpolate a value against a loaded percentile curve."""
    value = float(np.clip(value, x[0], x[-1]))
    idx = np.searchsorted(x, value)
    if idx >= len(x) - 1:
        return float(y[-1])

    x0, y0 = x[idx], y[idx]
    x1, y1 = x[idx + 1], y[idx + 1]
    return float(y0) if x1 == x0 else float((value - x0) / (x1 - x0) * (y1 - y0) + y0)


def get_weighted_mean(entries: list[ScoreEntry]) -> float:
    """
    Calculate the weighted mean of a list of ScoreEntries.
    Assumes entries are ordered by concept weight (e.g. [good, normal, bad] -> weights [2, 1, 0]).
    Uses the normalized_score of each entry to safely handle mixed bounds.
    """
    total = len(entries)
    weighted_mean = 0.0
    for i, entry in enumerate(entries):
        weighted_mean += (total - 1 - i) * entry.normalized_score
    return weighted_mean


def normalize_multiscore(
    entries: list[ScoreEntry],
    percentiles: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Extract a [0, 1] normalized summary score from a list of ScoreEntries."""
    if not entries:
        return 0.0

    weighted_mean = get_weighted_mean(entries)

    if percentiles is not None:
        x, y = percentiles
        return interp_percentile(weighted_mean, x, y)

    max_v = float(max(len(entries) - 1, 1))
    return float(np.clip((weighted_mean - 0.0) / max_v, 0.0, 1.0)) if max_v > 0.0 else 0.0
