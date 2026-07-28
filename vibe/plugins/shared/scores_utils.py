"""Shared utilities for normalizing and processing scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    x = np.concatenate(([0.0], x, [x[-1] + 1e-6])).astype(np.float32, copy=False)
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


def get_weighted_mean(
    scores: dict[int, float],
    label_map: dict[int, str] | None = None,
    label_order: list[str] | None = None,
) -> float:
    """Calculate the weighted mean of a score dictionary."""
    weighted_mean = 0.0

    if label_order is not None and label_map is not None:
        label_scores = {label_map[idx]: score for idx, score in scores.items()}
        ordered_values = [label_scores[label] for label in label_order if label in label_scores]
        total = len(ordered_values)
        for index, value in enumerate(ordered_values):
            weighted_mean += (total - 1 - index) * float(value)
        return weighted_mean

    # Sort by key to guarantee deterministic indexing regardless of backend dict insertion order
    for index, (_, value) in enumerate(sorted(scores.items())):
        weighted_mean += index * float(value)
    return weighted_mean


def normalize_multiscore(
    scores: dict[int, float],
    label_map: dict[int, str] | None = None,
    label_order: list[str] | None = None,
    percentiles: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Extract a [0, 1] normalized score from a multi-score distribution."""
    if not scores:
        return 0.0

    weighted_mean = get_weighted_mean(scores, label_map, label_order)

    if percentiles is not None:
        x, y = percentiles
        return interp_percentile(weighted_mean, x, y)

    max_v = float(max(len(scores) - 1, 1))
    return float(np.clip((weighted_mean - 0.0) / max_v, 0.0, 1.0)) if max_v > 0.0 else 0.0
