from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vibe.metadata import LabelCatalog, LabelInfo, ThresholdTable
from vibe.results import TagEntry, TagResult
from vibe.tag_categories import TagCategory

logger = logging.getLogger(__name__)


@dataclass
class ParsedTagData:
    """Compiled metadata resulting from parsing a tag CSV."""

    raw_tag_names: list[str]
    category_indices: dict[str, list[int]]
    catalog: LabelCatalog
    thresholds: ThresholdTable | None


def load_tag_metadata(
    csv_path: Path,
    category_labels: Mapping[int, TagCategory] | None = None,
    namespace: str = "unknown",
    *,
    threshold_column: str = "best_threshold",
) -> ParsedTagData:
    """
    Parse selected tag metadata from a CSV file and construct public data domain objects.
    """
    raw_tag_names: list[str] = []
    category_indices: dict[str, list[int]] = {}
    label_infos: list[LabelInfo] = []
    threshold_values: dict[str, float] = {}

    seen_categories: set[str] = set()
    category_labels_map = category_labels or {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_thresholds = threshold_column in (reader.fieldnames or [])

        for idx, row in enumerate(reader):
            raw_name = row.get("name")
            name = "" if raw_name is None else str(raw_name).strip()
            raw_tag_names.append(name)

            raw_category = row.get("category", "0")
            category_name = resolve_category_name(raw_category, category_labels_map, namespace=namespace)
            category_indices.setdefault(category_name, []).append(idx)
            seen_categories.add(category_name)

            raw_threshold = row.get(threshold_column, "") if has_thresholds else ""
            try:
                parsed_threshold = float(raw_threshold) if raw_threshold != "" else None
            except ValueError:
                parsed_threshold = None

            if parsed_threshold is not None:
                threshold_values[name] = parsed_threshold

            label_infos.append(
                LabelInfo(
                    index=idx,
                    name=name,
                    category=category_name,
                    threshold=parsed_threshold,
                )
            )

    catalog = LabelCatalog(
        labels=tuple(label_infos),
        categories=tuple(sorted(seen_categories)),
    )

    thresholds = None
    if threshold_values:
        thresholds = ThresholdTable(values=threshold_values, source="csv")

    return ParsedTagData(
        raw_tag_names=raw_tag_names,
        category_indices=category_indices,
        catalog=catalog,
        thresholds=thresholds,
    )


def resolve_category_name(
    raw_category: Any,
    category_labels: Mapping[int, TagCategory],
    *,
    namespace: str,
) -> str:
    """Resolve source category metadata to a canonical or custom category name."""
    raw_value = str(raw_category).strip()

    try:
        category_id = int(raw_value)
    except (TypeError, ValueError):
        category_id = None

    if category_id is not None:
        known_name = category_labels.get(category_id)
        if known_name is not None:
            return str(known_name)

    canonical_names = {str(category) for category in category_labels.values()}
    if raw_value in canonical_names:
        return raw_value

    return f"{namespace}:{raw_value or 'unknown'}"


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
        arr = np.ascontiguousarray(arr)  # resolve negative strides

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


def detect_is_logits(
    raw_output: Any,
    *,
    min_classes_for_prob_inference: int = 10,
) -> bool | None:
    """
    Heuristically determine whether model output represents raw logits or probabilities.

    Returns:
        - True: Values exist outside [0, 1] (mathematical proof of raw logits).
        - False: All values in [0, 1] with sufficient classes (strong proof of probabilities).
        - None: Ambiguous (e.g. few classes or scalar output where logits naturally fall in [0, 1]).
    """
    target = raw_output
    if isinstance(target, (tuple, list)):
        target = target[0] if target else target

    try:
        arr = np.asarray(target, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # 1. Definite logits: values outside [0, 1] cannot be probabilities
    if min_val < -1e-4 or max_val > 1.0 + 1e-4:
        return True

    # 2. Definite probabilities: in multi-class/multi-label taggers, unpredicted tags have strongly
    # negative logits (e.g. -5 to -15). It is practically impossible for all N > 10 classes to land in [0, 1].
    if arr.size >= min_classes_for_prob_inference:
        return False

    # 3. Ambiguous: small number of classes where logits could naturally be in [0, 1]
    return None


def normalize_output_scores(
    raw_output: Any,
    *,
    is_logits: bool,
    expected_count: int | None = None,
    audit_discrepancy: bool = True,
    use_softmax: bool = False,
) -> np.ndarray:
    """
    Flatten model output into probabilities in [0, 1].
    Audits the provided is_logits against detect_is_logits to warn about potential misconfigurations.
    Applies Softmax if use_softmax=True, otherwise applies independent Sigmoid.
    """
    if isinstance(raw_output, (tuple, list)):
        if len(raw_output) == 1:
            raw_output = raw_output[0]
        elif expected_count is not None:
            # 1. Try exact match on size or last dimension
            matching = None
            for item in raw_output:
                arr = np.asarray(item)
                if arr.size == expected_count or (arr.ndim > 0 and arr.shape[-1] == expected_count):
                    matching = item
                    break

            if matching is not None:
                raw_output = matching
            else:
                # 2. Fallback to closest size match + log warning
                candidates = [(abs(np.asarray(item).size - expected_count), item) for item in raw_output]
                candidates.sort(key=lambda x: x[0])
                closest_item = candidates[0][1]
                closest_shape = np.asarray(closest_item).shape

                logger.warning(
                    "No output tensor exactly matched expected_count=%d. "
                    "Selecting closest tensor with shape %s. Available shapes: %s",
                    expected_count,
                    closest_shape,
                    [np.asarray(x).shape for x in raw_output],
                )
                raw_output = closest_item
        else:
            logger.debug(
                "Model returned multiple outputs %s but no expected_count was provided. Defaulting to first output.",
                [np.asarray(x).shape for x in raw_output],
            )
            raw_output = raw_output[0]

    scores = np.asarray(raw_output, dtype=np.float32)

    if scores.ndim == 0:
        scores = scores.reshape(1)
    elif scores.ndim > 1:
        if scores.shape[0] == 1:
            scores = np.squeeze(scores, axis=0)
        scores = np.ravel(scores)

    # Discrepancy audit
    if audit_discrepancy and scores.size > 0:
        detected = detect_is_logits(scores)
        if detected is not None and detected != is_logits:
            if not is_logits and detected is True:
                logger.warning(
                    "normalize_output_scores was called with is_logits=False, but output contains "
                    "values outside [0, 1] (min=%.4f, max=%.4f). These appear to be raw logits.",
                    float(np.min(scores)),
                    float(np.max(scores)),
                )
            elif is_logits and detected is False:
                logger.warning(
                    "normalize_output_scores was called with is_logits=True, but all %d values are strictly "
                    "within [0, 1]. The output may already be normalized probabilities, which will cause "
                    "a double-activation distortion (compressing scores into [0.5, 0.73]).",
                    scores.size,
                )

    if is_logits:
        if use_softmax:
            shifted = scores - np.max(scores)
            exp_vals = np.exp(shifted)
            scores = exp_vals / np.sum(exp_vals)
        else:
            clipped = np.clip(scores, -80.0, 80.0)
            scores = 1.0 / (1.0 + np.exp(-clipped))
    else:
        scores = np.clip(scores, 0.0, 1.0)

    return scores.astype(np.float32, copy=False)


def build_categorized_tag_result(
    tag_names: list[str],
    scores: np.ndarray,
    category_indices: dict[str, list[int]],
) -> TagResult:
    """Safely build a TagResult from raw arrays using pre-mapped category indices."""
    usable_count = min(len(scores), len(tag_names))
    result_categories: dict[str, list[TagEntry]] = {}

    for cat_name, indices in category_indices.items():
        if not indices:
            continue

        entries: list[TagEntry] = []
        for idx in indices:
            if idx >= usable_count:
                continue
            entries.append(TagEntry(tag=tag_names[idx], score=float(scores[idx])))

        entries.sort(key=lambda item: item.score, reverse=True)
        result_categories[cat_name] = entries

    return TagResult(categories=result_categories)


def _to_rgb_with_background(image: Any) -> Any:
    """Convert image to RGB and flatten alpha over a white background."""
    from PIL import Image

    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background

    if image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    return image.convert("RGB")
