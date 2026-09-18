"""Shared utilities for normalizing and processing scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_scalar(score: float, score_min: float, score_max: float) -> float:
    """Normalize a scalar score to a strict [0, 1] range."""
    if score_max <= score_min:
        return 0.0
    return float(np.clip((score - score_min) / (score_max - score_min), 0.0, 1.0))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along specified axis with numerical stability."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


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
