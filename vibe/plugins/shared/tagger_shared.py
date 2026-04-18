from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vibe.results import TagEntry
from vibe.tag_categories import DanbooruTagCategory


@dataclass
class TagMetadata:
    """Parsed tag metadata loaded from selected_tags.csv-like files."""

    raw_tag_names: list[str] = field(default_factory=list)
    category_indices: dict[int, list[int]] = field(default_factory=dict)
    per_tag_thresholds: list[float | None] = field(default_factory=list)

    def indices_for(self, category: int) -> list[int]:
        """Return CSV row indices for a category, or an empty list when absent."""
        return self.category_indices.get(category, [])


def load_tag_metadata(
    csv_path: Path,
    *,
    threshold_column: str = "best_threshold",
) -> TagMetadata:
    """
    Parse selected tag metadata from a CSV file.

    The parser is intentionally permissive: missing/invalid columns fall back
    to safe defaults so models with reduced metadata can still run.
    """
    metadata = TagMetadata()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            metadata.raw_tag_names.append(row.get("name", ""))

            try:
                category = int(row.get("category", "0"))
            except ValueError:
                category = int(DanbooruTagCategory.GENERAL)
            metadata.category_indices.setdefault(category, []).append(idx)

            raw_threshold = row.get(threshold_column, "")
            try:
                parsed_threshold = float(raw_threshold) if raw_threshold != "" else None
            except ValueError:
                parsed_threshold = None
            metadata.per_tag_thresholds.append(parsed_threshold)

    return metadata


def preprocess_tagger_image(
    image: Any,
    *,
    image_size: int,
    input_layout: str = "NHWC",
    rgb_to_bgr: bool = False,
    normalize_to_unit: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Convert image-like input into a float32 model-ready batch tensor."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image = _to_rgb_with_background(image)

    width, height = image.size
    if width != height:
        size = max(width, height)
        squared = Image.new("RGB", (size, size), (255, 255, 255))
        squared.paste(image, ((size - width) // 2, (size - height) // 2))
        image = squared

    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    if rgb_to_bgr:
        arr = arr[:, :, ::-1]

    if normalize_to_unit:
        arr = arr / 255.0

    if mean is not None and std is not None:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean_arr) / std_arr

    if input_layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))
    elif input_layout != "NHWC":
        raise ValueError(f"Unsupported input_layout '{input_layout}'. Expected 'NHWC' or 'NCHW'.")

    return np.expand_dims(arr, axis=0).astype(np.float32, copy=False)


def normalize_output_scores(raw_output: Any) -> np.ndarray:
    """Flatten model output into probabilities in [0, 1]."""
    scores = np.asarray(raw_output, dtype=np.float32)

    if scores.ndim == 0:
        scores = scores.reshape(1)
    elif scores.ndim > 1:
        if scores.shape[0] == 1:
            scores = np.squeeze(scores, axis=0)
        scores = np.ravel(scores)

    if np.min(scores) < 0.0 or np.max(scores) > 1.0:
        clipped = np.clip(scores, -80.0, 80.0)
        scores = 1.0 / (1.0 + np.exp(-clipped))

    return scores.astype(np.float32, copy=False)


def build_entries_for_indices(
    *,
    tag_names: list[str],
    indices: list[int],
    scores: np.ndarray,
    usable_count: int,
    thresholds: list[float | None] | None = None,
    entry_factory: Callable[[str, float, float | None], TagEntry] | None = None,
) -> list[TagEntry]:
    """Build score-sorted entries for a category index list."""
    entries: list[TagEntry] = []
    for idx in indices:
        if idx >= usable_count:
            continue
        tag = tag_names[idx]
        score = float(scores[idx])
        threshold = thresholds[idx] if thresholds is not None and idx < len(thresholds) else None
        if entry_factory is None:
            entries.append(TagEntry(tag=tag, score=score))
        else:
            entries.append(entry_factory(tag, score, threshold))

    entries.sort(key=lambda item: item.score, reverse=True)
    return entries


def _to_rgb_with_background(image: Any) -> Any:
    """Convert image to RGB and flatten alpha over a white background."""
    from PIL import Image

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background

    if image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background

    return image.convert("RGB")
