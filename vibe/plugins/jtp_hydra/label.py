import random
from collections import defaultdict
from functools import cache
from itertools import chain
from sys import intern
from typing import Callable, Iterable, TextIO

from torch import Tensor

__all__ = ("TAG_CATEGORIES", "Label", "Rewriter", "parse_aliases")

TAG_CATEGORIES = [
    "general",
    "artist",
    "contributor",
    "copyright",
    "character",
    "species",
    "invalid",
    "meta",
    "lore",
]

COLOR_PREFIXES = (
    "black_",
    "blue_",
    "brown_",
    "green_",
    "grey_",
    "orange_",
    "pink_",
    "purple_",
    "red_",
    "tan_",
    "teal_",
    "white_",
    "yellow_",
    "dark_",
    "light_",
    "blonde_",
)

COLOR_EXCEPTIONS = (
    "_background",
    "_border",
    "_outline",
    "black_bars",
    "dark_room",
    "light_beam",
    "light_bulb",
    "tan_line",
)


class Label:
    __slots__ = ("label", "category", "implies", "validation")

    def __init__(
        self,
        label: str,
        category: str,
        implies: Iterable[str] = (),
        validation: Tensor | None = None,
    ):
        self.label = intern(label)
        self.category = intern(category)
        self.implies = [intern(impl) for impl in implies]
        self.validation = validation

    def __str__(self) -> str:
        return self.label

    @cache
    def is_color(self) -> bool:
        return (
            self.category in ("", "general")
            and self.label.startswith(COLOR_PREFIXES)
            and not self.label.endswith(COLOR_EXCEPTIONS)
        )


class Rewriter(dict[str, str]):
    __slots__ = ("rewrite_fn", "separator")

    def __init__(
        self,
        rewrite_fn: Callable[[Label], str],
        separator: str = ", ",
        labels: Iterable[Label] | None = None,
        *,
        check: bool = True,
    ) -> None:
        self.rewrite_fn = rewrite_fn
        self.separator = separator

        if labels is not None:
            self.add(labels)

            if check:
                self.check()

    def add(self, labels: Iterable[Label]) -> None:
        for label in labels:
            if label.label not in self:
                self[label.label] = self.rewrite_fn(label)

    def check(self) -> None:
        if conflicts := self.conflicts(self.items()):
            raise ValueError(
                "Conflicting rewritten tags: "
                + "; ".join(" ".join(repr(key) for key in keys) + f" > {repr(value)}" for keys, value in conflicts)
            )

    def rewrite(self, labels: Iterable[str]) -> Iterable[str]:
        for label in labels:
            yield self[label]

    def rewrite_join(self, labels: Iterable[str], *, prefix: str | None = "", shuffle: bool = False) -> str:
        return self.join(self.rewrite(labels), sep=self.separator, prefix=prefix, shuffle=shuffle)

    @staticmethod
    def join(labels: Iterable[str], *, sep: str = ", ", prefix: str | None = "", shuffle: bool = False):
        if shuffle:
            labels = list(labels)
            random.shuffle(labels)

        if prefix:
            prefix_labels = [label.strip() for label in prefix.split(sep.strip() or sep) if label.strip()]
            if prefix_labels:
                labels = chain(prefix_labels, filter(lambda label: label not in prefix_labels, labels))

        return str.join(sep, labels)

    @staticmethod
    def create(
        labels: Iterable[Label] | None = None,
        *,
        aliases: dict[str, str] | str | None = None,
        prefixes: dict[str, str] | str | None = None,
        separator: str | None = None,
        spaces: bool = True,
        escape: bool = False,
    ) -> "Rewriter":
        if aliases is None:
            aliases = {}
        elif isinstance(aliases, str):
            aliases = dict(parse_aliases(aliases))

        if prefixes is None:
            prefixes = {}
        elif isinstance(prefixes, str):
            prefixes = dict(parse_aliases(prefixes))

        trans: dict[int, str] = {}
        if spaces:
            trans[ord("_")] = " "

        if escape:
            trans[ord("(")] = r"\("
            trans[ord(")")] = r"\)"
            trans[ord(":")] = r"\:"

        def rewrite(label: Label) -> str:
            rewritten = label.label

            if (alias := aliases.get(rewritten)) is not None:
                rewritten = alias

            if (prefix := prefixes.get(label.category)) is not None:
                rewritten = prefix + rewritten

            if trans:
                rewritten = rewritten.translate(trans)

            return rewritten

        if separator is None:
            separator = ", " if spaces else " "

        return Rewriter(rewrite, separator, labels)

    @staticmethod
    def conflicts(mapping: Iterable[tuple[str, str]]) -> list[tuple[list[str], str]]:
        reverse: dict[str, list[str]] = defaultdict(list)
        conflicts: list[tuple[list[str], str]] = []
        for k, v in mapping:
            bucket = reverse[v]
            bucket.append(k)
            if len(bucket) == 2:
                conflicts.append((bucket, v))

        return conflicts


def parse_aliases(value: TextIO | str) -> Iterable[tuple[str, str]]:
    if not isinstance(value, str):
        value = value.read()

    for idx, line in enumerate(value.splitlines()):
        line, _, _ = line.partition("#")

        parts = line.split()
        match len(parts):
            case 0:
                pass
            case 2:
                yield tuple(parts)
            case _:
                raise ValueError(f"Invalid alias on line {idx + 1}: {repr(line)}")
