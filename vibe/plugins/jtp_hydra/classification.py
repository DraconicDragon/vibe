import re
from dataclasses import dataclass
from functools import cache
from itertools import islice
from typing import Callable, Iterable, Sequence, TextIO, TypeAlias

import torch
from torch import Tensor

from .label import Label

__all__ = (
    "IMPLICATION_MODES",
    "DEFAULT_METRIC",
    "Metric",
    "MetricLike",
    "Operator",
    "Calibration",
    "ExclusiveGroup",
    "FilterLabels",
    "FilterResults",
    "QualifyResults",
    "SortResults",
    "InheritImplications",
    "ConstrainImplications",
    "RemoveImplications",
    "min_pr",
    "f_score",
    "csi",
    "parse_metric",
)

IMPLICATION_MODES = ("inherit", "constrain", "remove", "constrain-remove", "off")

Metric: TypeAlias = Callable[[float, float, float, float], float | None]
MetricLike: TypeAlias = Metric | Iterable[float] | float | str
Operator: TypeAlias = Callable[["Calibration", dict[str, float]], None]


def min_pr(inner: Metric, min_precision: float = 0.0, min_recall: float = 0.0) -> Metric:
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be between 0.0 and 1.0, inclusive")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be between 0.0 and 1.0, inclusive")

    if min_precision == 0.0 and min_recall == 0.0:
        return inner

    return lambda tp, fp, tn, fn: (
        inner(tp, fp, tn, fn) if (tp and tp / (tp + fp) >= min_precision and tp / (tp + fn) >= min_recall) else None
    )


def f_score(beta: float = 1.0) -> Metric:
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    w_fn = beta * beta
    w_tp = 1.0 + w_fn
    return lambda tp, fp, tn, fn: (tp * w_tp) / (tp * w_tp + fp + fn * w_fn) if tp else 0.0


def csi(w_fp: float = 1.0) -> Metric:
    if w_fp <= 0.0:
        raise ValueError("w_fp must be positive")

    return lambda tp, fp, tn, fn: tp / (tp + fp * w_fp + fn) if tp else 0.0


DEFAULT_METRIC = min_pr(f_score(1.0), min_precision=0.1)

_EMPTY: frozenset = frozenset()


def log_interval(value: float, scale: float) -> float:
    assert 0.0 <= value <= 1.0
    assert scale > 0.0

    return scale ** (1.0 - 2 * value)


def simple_slider(
    value: float, min_precision: float = 0.1, metric: Callable[[float], Metric] = f_score, scale: float = 4.0
) -> Metric:
    return min_pr(metric(log_interval(value, scale)), min_precision=min_precision)


def parse_metric(value: str) -> Metric:
    if value == "default":
        return DEFAULT_METRIC

    if (match := re.fullmatch(r"(f|csi)([0-9]+\.[0-9]+)?(?:@(0\.[0-9]+))?", value)) is None:
        raise ValueError("Invalid metric.")

    metric, arg_s, prec_s = match.groups()

    arg = float(arg_s) if arg_s else 1.0
    prec = float(prec_s) if prec_s else 0.0

    metric_fn: Callable[[float], Metric]
    match metric:
        case "f":
            metric_fn = f_score
        case "csi":
            metric_fn = csi
        case _:
            raise AssertionError

    return min_pr(metric_fn(arg), min_precision=prec)


class Calibration(dict[str, tuple[Label, float]]):
    __slots__ = ()

    def __init__(self, labels: Iterable[Label], metric: MetricLike) -> None:
        if callable(metric):
            super().__init__((label.label, (label, self.calibrate_label(label, metric)[0])) for label in labels)
        elif isinstance(metric, str):
            metric = parse_metric(metric)
            super().__init__((label.label, (label, self.calibrate_label(label, metric)[0])) for label in labels)
        elif isinstance(metric, float):
            super().__init__((label.label, (label, metric)) for label in labels)
        else:
            super().__init__((label.label, (label, threshold)) for label, threshold in zip(labels, metric, strict=True))

    def __setitem__(self, key: str, value: Metric | float | None) -> None:  # type: ignore[override]
        label = self[key][0]

        if value is None:
            value = float("inf")
        elif callable(value):
            value = self.calibrate_label(label, value)[0]

        super().__setitem__(label.label, (label, value))

    def __delitem__(self, key: str) -> None:
        super().__setitem__(key, (self[key][0], float("inf")))

    def labels(self) -> Iterable[Label]:
        for label, _ in self.values():
            yield label

    def thresholds(self) -> Iterable[float]:
        for _, threshold in self.values():
            yield threshold

    def as_thresholds(self) -> list[float]:
        return list(self.thresholds())

    def get_threshold(self, label: str) -> float:
        return self[label][1]

    def classify_output(
        self,
        output: Tensor,
        *,
        implications: str = "off",
        exclude_labels: set[str] | frozenset[str] | None = _EMPTY,
        exclude_categories: set[str] | frozenset[str] | None = _EMPTY,
        exclusive_groups: Iterable["ExclusiveGroup"] = (),
        sort: bool = True,
    ) -> dict[str, float]:
        return self.apply_to_output(
            output,
            self.default_operators(
                implications=implications,
                exclude_labels=exclude_labels,
                exclude_categories=exclude_categories,
                pre_qualify=exclusive_groups,
                sort=sort,
            ),
        )

    def classify_outputs(
        self,
        outputs: Tensor,
        *,
        implications: str = "off",
        exclude_labels: set[str] | frozenset[str] | None = _EMPTY,
        exclude_categories: set[str] | frozenset[str] | None = _EMPTY,
        exclusive_groups: Sequence["ExclusiveGroup"] = (),
        sort: bool = True,
    ) -> list[dict[str, float]]:
        return list(
            self.apply_to_outputs(
                outputs,
                list(
                    self.default_operators(
                        implications=implications,
                        exclude_labels=exclude_labels,
                        exclude_categories=exclude_categories,
                        pre_qualify=exclusive_groups,
                        sort=sort,
                    )
                ),
            )
        )

    def apply_to_output(self, output: Tensor, operators: Iterable[Operator]) -> dict[str, float]:
        if output.shape != (len(self),):
            raise ValueError(f"Output tensor has shape {output.shape}, not ({len(self)}).")

        results = {label: prob for label, prob in zip(self.keys(), output.tolist())}

        for op in operators:
            op(self, results)

        return results

    def apply_to_outputs(self, outputs: Tensor, operators: Sequence[Operator]) -> Iterable[dict[str, float]]:
        match outputs.ndim:
            case 1:
                yield self.apply_to_output(outputs, operators)
            case 2:
                for output in outputs.unbind(0):
                    yield self.apply_to_output(output, operators)
            case _:
                raise ValueError(f"Outputs tensor has {outputs.ndim} dimensions, but expected 1 or 2.")

    @staticmethod
    def default_operators(
        implications: str = "off",
        exclude_labels: set[str] | frozenset[str] | None = frozenset(),
        exclude_categories: set[str] | frozenset[str] | None = frozenset(),
        filter_fn: Callable[[Label], bool] | None = None,
        pre_qualify: Iterable[Operator] = (),
        post_qualify: Iterable[Operator] = (),
        sort: bool = False,
    ) -> Iterable[Operator]:
        match implications:
            case "inherit":
                yield InheritImplications
            case "constrain" | "constrain-remove":
                yield ConstrainImplications
            case "remove" | "off":
                pass
            case _:
                raise ValueError("Invalid implications mode.")

        yield from pre_qualify

        if exclude_labels or exclude_categories or filter_fn is not None:
            if exclude_labels is None:
                exclude_labels = _EMPTY
            if exclude_categories is None:
                exclude_categories = _EMPTY
            if filter_fn is None:

                def filter_fn(_):
                    return True

            exclude_colors = "color" in exclude_categories

            yield FilterResults(
                lambda label: (
                    label.category not in exclude_categories
                    and label.label not in exclude_labels
                    and not (exclude_colors and label.is_color())
                    and filter_fn(label)
                )
            )
        else:
            yield QualifyResults

        yield from post_qualify

        if implications in ("remove", "constrain-remove"):
            yield RemoveImplications

        if sort:
            yield SortResults

    @classmethod
    def calibrate_labels(cls, labels: Iterable[Label], metric: Metric) -> Iterable[float]:
        for label in labels:
            yield cls.calibrate_label(label, metric)[0]

    @staticmethod
    def calibrate_label(label: Label, metric: Metric) -> tuple[float, float | None]:
        if label.validation is None:
            raise ValueError(f"Label {repr(label.label)} is missing validation data for calibration.")

        return Calibration.calibrate(label.validation, metric)

    @staticmethod
    def calibrate(validation: Tensor, metric: Metric) -> tuple[float, float | None]:
        best_idx: int | None = None
        best_score: float | None = None
        for idx in range(validation.size(0)):
            score = metric(*validation[idx].tolist())
            if score is not None and (best_score is None or score > best_score):
                best_idx = idx
                best_score = score

        if best_idx is None:
            return float("inf"), None

        best_threshold = (best_idx + 1) / (validation.size(0) + 1)

        return Calibration.bf16_threshold(best_threshold), best_score

    @cache
    @staticmethod
    def bf16_threshold(value: float) -> float:
        return (
            torch.tensor(value, device="cpu", dtype=torch.float32)
            .logit_()
            .to(dtype=torch.bfloat16)
            .to(dtype=torch.float32)
            .sigmoid_()
            .item()
        )


@dataclass(eq=False, match_args=False, slots=True)
class ExclusiveGroup:
    group: Sequence[str]
    requirements: Sequence[str] = ()
    exceptions: Sequence[str] = ()

    def __call__(self, calibration: "Calibration", results: dict[str, float]) -> None:
        for requirement in self.requirements:
            if results.get(requirement, float("-inf")) < calibration.get_threshold(requirement):
                return

        for exception in self.exceptions:
            if results.get(exception, float("-inf")) >= calibration.get_threshold(exception):
                return

        best_label: str | None = None
        best_prob = float("-inf")
        for label in self.group:
            if (prob := results.get(label)) is None:
                continue

            if prob <= best_prob:
                del results[label]
                _remove_antecedents(results, label, calibration)
                continue

            if best_label is not None:
                results.pop(best_label, None)  # malformed group could have removed best label
                _remove_antecedents(results, best_label, calibration)

            best_label = label
            best_prob = prob

    @staticmethod
    def parse(value: Sequence[str] | str) -> "ExclusiveGroup":
        parts = value.split() if isinstance(value, str) else value

        split_idx = 0
        for idx, part in enumerate(parts):
            if part.endswith(":"):
                split_idx = idx + 1
            elif not part:
                raise ValueError(f"Item {idx + 1} is empty.")

        requirements: list[str] = []
        exceptions: list[str] = []
        if split_idx:
            for idx, part in enumerate(islice(parts, split_idx)):
                if part.endswith(":"):
                    if len(part) == 1:
                        continue

                    part = part[:-1]

                if part.startswith("!"):
                    if len(part) == 1:
                        raise ValueError(f"Item {idx + 1} is empty.")

                    exceptions.append(part[1:])
                else:
                    requirements.append(part)

        group: list[str] = []
        for part in islice(parts, split_idx, None):
            if part not in group:
                group.append(part)

        if len(group) < 2:
            raise ValueError("At least two distinct items are required.")

        return ExclusiveGroup(group, requirements, exceptions)

    @staticmethod
    def parse_file(value: TextIO | str) -> Iterable["ExclusiveGroup"]:
        if not isinstance(value, str):
            value = value.read()

        for idx, line in enumerate(value.splitlines()):
            line = re.split(r"(?:^| |\t)#", line, maxsplit=1)[0]
            if parts := line.split():
                try:
                    yield ExclusiveGroup.parse(parts)
                except ValueError as ex:
                    raise ValueError(f"Invalid group on line {idx + 1}: {ex}") from ex


class FilterLabels:
    def __init__(self, filter_fn: Callable[[Label], bool]) -> None:
        self.filter_fn = filter_fn

    def __call__(self, calibration: Calibration, results: dict[str, float]) -> None:
        for label in calibration.labels():
            if label.label in results and not self.filter_fn(label):
                del results[label.label]


class FilterResults:
    def __init__(self, filter_fn: Callable[[Label], bool]) -> None:
        self.filter_fn = filter_fn

    def __call__(self, calibration: Calibration, results: dict[str, float]) -> None:
        for label, threshold in calibration.values():
            if (prob := results.get(label.label)) is None:
                continue

            if prob < threshold or not self.filter_fn(label):
                del results[label.label]


def QualifyResults(calibration: Calibration, results: dict[str, float]) -> None:
    for label, threshold in calibration.values():
        if (prob := results.get(label.label)) is not None and prob < threshold:
            del results[label.label]


def InheritImplications(calibration: Calibration, results: dict[str, float]) -> None:
    for result in results:
        _inherit_implications(results, result, calibration)


def ConstrainImplications(calibration: Calibration, results: dict[str, float]) -> None:
    for result in results:
        _constrain_implications(results, result, result, calibration)


def RemoveImplications(calibration: Calibration, results: dict[str, float]) -> None:
    for result in list(results.keys()):
        _remove_consequents(results, result, calibration)


def SortResults(calibration: Calibration, results: dict[str, float]) -> None:
    values = sorted(results.items(), key=lambda item: (-item[1], item[0]))
    results.clear()
    results.update(values)


def _inherit_implications(outputs: dict[str, float], antecedent: str, labels: dict[str, tuple[Label, float]]) -> None:
    p = outputs[antecedent]

    if (label := labels.get(antecedent)) is None:
        return

    for consequent in label[0].implies:
        if (q := outputs.get(consequent)) is None:
            continue

        if q < p:
            outputs[consequent] = p

        _inherit_implications(outputs, consequent, labels)


def _constrain_implications(
    outputs: dict[str, float], target: str, antecedent: str, labels: dict[str, tuple[Label, float]]
) -> None:
    if (label := labels.get(antecedent)) is None:
        return

    for consequent in label[0].implies:
        if (p := outputs.get(consequent)) is None:
            continue

        if outputs[target] > p:
            outputs[target] = p

        _constrain_implications(outputs, target, consequent, labels)


def _remove_antecedents(outputs: dict[str, float], consequent: str, labels: dict[str, tuple[Label, float]]) -> None:
    for label, _ in labels.values():
        if consequent in label.implies:
            outputs.pop(label.label, None)
            _remove_antecedents(outputs, label.label, labels)


def _remove_consequents(outputs: dict[str, float], antecedent: str, labels: dict[str, tuple[Label, float]]) -> None:
    if (label := labels.get(antecedent)) is None:
        return

    for consequent in label[0].implies:
        outputs.pop(consequent, None)
        _remove_consequents(outputs, consequent, labels)
