from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# Actual JTP/Hydra E621 category IDs.
CATEGORY_IDS = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "species",
    7: "meta",
    8: "lore",
}


def hydra_threshold_grid(n: int) -> np.ndarray:
    """
    Reproduce the threshold construction used by the upstream calibrator.

    The values are:
        linspace(0, 1, n + 2)[1:-1]
        -> logit
        -> bfloat16
        -> float32
        -> sigmoid

    This means the resulting values are not exactly 0.01, 0.02, ...
    """
    thresholds = (
        torch.linspace(
            0.0,
            1.0,
            n + 2,
            dtype=torch.float32,
        )[1:-1]
        .logit()
        .to(dtype=torch.bfloat16)
        .to(dtype=torch.float32)
        .sigmoid()
    )

    return thresholds.numpy()


def calculate_metrics(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate precision, recall and F1 for every threshold."""
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fp) > 0,
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(tp + fn) > 0,
    )

    f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(2.0 * tp + fp + fn) > 0,
    )

    return precision, recall, f1


def choose_threshold(
    thresholds: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    *,
    min_precision: float,
) -> dict[str, float] | None:
    """
    Choose the threshold with maximum F1 subject to minimum precision.

    Ties are resolved in favor of the lower threshold, matching the
    upstream calibrator's comparison:
        (score, -threshold)
    """
    precision, recall, f1 = calculate_metrics(tp, fp, fn)

    valid = precision >= min_precision

    if not np.any(valid):
        return None

    candidate_f1 = np.where(valid, f1, -np.inf)

    # np.argmax returns the first maximum, and thresholds are ascending.
    index = int(np.argmax(candidate_f1))

    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "tp": float(tp[index]),
        "fp": float(fp[index]),
        "fn": float(fn[index]),
    }


def print_result(name: str, result: dict[str, float] | None) -> None:
    if result is None:
        print(f"{name:<12} no threshold satisfies precision >= configured minimum")
        return

    print(
        f"{name:<12} "
        f"threshold={result['threshold']:.4f}  "
        f"precision={result['precision']:.4f}  "
        f"recall={result['recall']:.4f}  "
        f"F1={result['f1']:.4f}"
    )


def load_jtp3(
    val_path: Path,
    tags_path: Path,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """
    Load JTP-3 validation CSV.

    Returns:
        thresholds
        groups: category -> (TP, FP, FN)
        unmatched_tag_count

    Only tags present in the model metadata are included.
    """
    tag_categories: dict[str, str] = {}

    with tags_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fields = set(reader.fieldnames or ())
        required = {"tag", "category"}

        missing = required - fields
        if missing:
            raise RuntimeError(f"{tags_path} is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            tag = row["tag"].strip()
            if not tag:
                continue

            try:
                category_id = int(row["category"])
            except ValueError as exc:
                raise RuntimeError(f"Invalid category for tag {tag!r}: {row['category']!r}") from exc

            category = CATEGORY_IDS.get(
                category_id,
                f"category_{category_id}",
            )

            tag_categories[tag] = category

    # tag -> threshold -> [tp, fp, fn]
    per_tag = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))

    unmatched_tags: set[str] = set()

    with val_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fields = set(reader.fieldnames or ())
        required = {"tag", "threshold", "tp", "fp", "tn", "fn"}

        missing = required - fields
        if missing:
            raise RuntimeError(f"{val_path} is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            tag = row["tag"].strip()

            if not tag:
                continue

            if tag not in tag_categories:
                unmatched_tags.add(tag)
                continue

            threshold = float(row["threshold"])
            tp = float(row["tp"])
            fp = float(row["fp"])
            fn = float(row["fn"])

            values = per_tag[tag][threshold]
            values[0] += tp
            values[1] += fp
            values[2] += fn

    if not per_tag:
        raise RuntimeError("No validation tags matched the model metadata.")

    # All JTP-3 validation rows should share the same threshold grid.
    threshold_values = sorted({threshold for tag_data in per_tag.values() for threshold in tag_data})

    thresholds = np.asarray(
        threshold_values,
        dtype=np.float64,
    )

    groups: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    # Global: all model tags.
    global_tp = np.zeros(len(thresholds), dtype=np.float64)
    global_fp = np.zeros(len(thresholds), dtype=np.float64)
    global_fn = np.zeros(len(thresholds), dtype=np.float64)

    category_accumulator = defaultdict(
        lambda: [
            np.zeros(len(thresholds), dtype=np.float64),
            np.zeros(len(thresholds), dtype=np.float64),
            np.zeros(len(thresholds), dtype=np.float64),
        ]
    )

    threshold_index = {threshold: index for index, threshold in enumerate(thresholds)}

    for tag, threshold_data in per_tag.items():
        category = tag_categories[tag]

        category_tp, category_fp, category_fn = category_accumulator[category]

        for threshold, (tp, fp, fn) in threshold_data.items():
            index = threshold_index[threshold]

            global_tp[index] += tp
            global_fp[index] += fp
            global_fn[index] += fn

            category_tp[index] += tp
            category_fp[index] += fp
            category_fn[index] += fn

    groups["global"] = (
        global_tp,
        global_fp,
        global_fn,
    )

    for category, (
        category_tp,
        category_fp,
        category_fn,
    ) in sorted(category_accumulator.items()):
        groups[category] = (
            category_tp,
            category_fp,
            category_fn,
        )

    return thresholds, groups, len(unmatched_tags)


def load_hydra35(
    weights_path: Path,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load Hydra 3.5's embedded validation tensor.

    Expected tensor:
        validation.shape == (num_tags, num_thresholds, 4)

    Columns:
        0 = TP
        1 = FP
        2 = TN
        3 = FN
    """
    with safe_open(
        str(weights_path),
        framework="numpy",
    ) as file:
        metadata = file.metadata() or {}

        labels_raw = metadata.get("classifier.labels")
        if labels_raw is None:
            raise RuntimeError("Hydra model metadata has no 'classifier.labels'.")

        tags: list[str] = []
        categories: list[str] = []

        for index, line in enumerate(labels_raw.splitlines()):
            line = line.strip()

            if not line:
                tags.append(f"unknown_{index}")
                categories.append("unknown")
                continue

            parts = line.split(maxsplit=2)

            tag = parts[0]
            category_raw = parts[1] if len(parts) > 1 else "general"

            try:
                category = CATEGORY_IDS.get(
                    int(category_raw),
                    f"category_{category_raw}",
                )
            except ValueError:
                category = category_raw

            tags.append(tag)
            categories.append(category)

        if "validation" not in file.keys():  # noqa: SIM118
            raise RuntimeError("Hydra model has no embedded 'validation' tensor.")

        validation = np.asarray(file.get_tensor("validation"))

    if validation.ndim != 3 or validation.shape[2] < 4:
        raise RuntimeError(f"Unexpected validation tensor shape: {validation.shape}; expected (tags, thresholds, 4).")

    if validation.shape[0] != len(tags):
        raise RuntimeError(
            f"Validation contains {validation.shape[0]} tags, but classifier.labels contains {len(tags)}."
        )

    n_thresholds = validation.shape[1]

    thresholds = hydra_threshold_grid(n_thresholds)

    tp_all = validation[:, :, 0].astype(np.float64)
    fp_all = validation[:, :, 1].astype(np.float64)
    fn_all = validation[:, :, 3].astype(np.float64)

    groups: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    # Global.
    groups["global"] = (
        tp_all.sum(axis=0),
        fp_all.sum(axis=0),
        fn_all.sum(axis=0),
    )

    # Per category.
    category_indices: dict[str, list[int]] = defaultdict(list)

    for index, category in enumerate(categories):
        category_indices[category].append(index)

    for category, indices in sorted(category_indices.items()):
        indices_array = np.asarray(indices)

        groups[category] = (
            tp_all[indices_array].sum(axis=0),
            fp_all[indices_array].sum(axis=0),
            fn_all[indices_array].sum(axis=0),
        )

    return thresholds, groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Calculate derived global and per-category thresholds from JTP-3 / Hydra validation data.")
    )

    parser.add_argument(
        "--model",
        choices=("jtp3", "hydra"),
        required=True,
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Model safetensors file.",
    )

    parser.add_argument(
        "--val",
        type=Path,
        help="JTP-3 validation CSV.",
    )

    parser.add_argument(
        "--tags",
        type=Path,
        help="JTP-3 tag metadata CSV.",
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.1,
        help="Minimum precision constraint. Default: 0.1",
    )

    args = parser.parse_args()

    if not 0.0 <= args.min_precision <= 1.0:
        parser.error("--min-precision must be between 0 and 1.")

    if args.model == "jtp3":
        if args.val is None:
            parser.error("--val is required for --model jtp3")

        if args.tags is None:
            parser.error("--tags is required for --model jtp3")

        thresholds, groups, unmatched_count = load_jtp3(
            args.val,
            args.tags,
        )

    else:
        thresholds, groups = load_hydra35(
            args.weights,
        )
        unmatched_count = 0

    print(f"Candidate thresholds: {len(thresholds)}")
    print(f"Range: {thresholds[0]:.6f} .. {thresholds[-1]:.6f}")
    print(f"Minimum precision: {args.min_precision:.4f}")

    if args.model == "jtp3":
        print(f"Validation tags excluded because they are absent from model metadata: {unmatched_count}")

    print()
    print("Recommended thresholds")
    print("======================")

    result = choose_threshold(
        thresholds,
        *groups["global"],
        min_precision=args.min_precision,
    )

    print_result("GLOBAL", result)

    print()
    print("Per category")
    print("============")

    for category in sorted(groups):
        if category == "global":
            continue

        result = choose_threshold(
            thresholds,
            *groups[category],
            min_precision=args.min_precision,
        )

        print_result(category, result)


if __name__ == "__main__":
    main()
