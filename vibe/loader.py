"""File loader and source string resolution for model assets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

from vibe.backends.base import ArtifactMap, ModelVariant
from vibe.hf_downloader import HFDownloadError, download_or_cached_with_reason

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised when file resolution or validation fails."""


class SourceResolver(ABC):
    @abstractmethod
    def resolve(self, variant: ModelVariant, **kwargs) -> ArtifactMap:
        pass


class LocalResolver(SourceResolver):
    def __init__(self, folder: Path, file_name_map: Mapping[str, str] | None = None):
        self.folder = folder
        self.file_name_map = file_name_map or {}

    def resolve(self, variant: ModelVariant, **kwargs) -> ArtifactMap:
        if not self.folder.is_dir():
            raise LoaderError(f"Local folder does not exist: {self.folder}")

        paths: dict[str, Path] = {}
        optional_missing: dict[str, str] = {}

        for spec in variant.artifacts:
            # Respect explicit filename overrides mapped via ID, else use default name
            mapped_name = self.file_name_map.get(spec.id, spec.name)
            candidate = self.folder / mapped_name

            if candidate.is_file():
                paths[spec.id] = candidate
            elif spec.required:
                raise LoaderError(f"Required artifact '{spec.id}' ({mapped_name}) not found in {self.folder}")
            else:
                optional_missing[spec.id] = f"Missing in local folder {self.folder}"

        return ArtifactMap(paths, optional_missing)


class HFResolver(SourceResolver):
    def __init__(self, repo_id: str, file_name_map: Mapping[str, str] | None = None):
        self.repo_id = repo_id
        self.file_name_map = file_name_map or {}

    def resolve(self, variant: ModelVariant, **kwargs) -> ArtifactMap:
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        allow_download = kwargs.get("allow_download", True)

        paths: dict[str, Path] = {}
        optional_missing: dict[str, str] = {}

        # The variant provides the baseline repo_id, but the user string can override it.
        base_repo = self.repo_id or variant.repo_id

        for spec in variant.artifacts:
            mapped_name = self.file_name_map.get(spec.id, spec.name)
            if variant.hf_subdir:
                mapped_name = f"{variant.hf_subdir.rstrip('/')}/{mapped_name}"

            try:
                local_path, reason = download_or_cached_with_reason(
                    repo_id=base_repo,
                    filename=mapped_name,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_download=allow_download,
                    required=spec.required,
                )
                if local_path:
                    paths[spec.id] = Path(local_path)
                elif not spec.required and reason:
                    optional_missing[spec.id] = reason

            except HFDownloadError as exc:
                raise LoaderError(str(exc)) from None

        return ArtifactMap(paths, optional_missing)


class AutoResolver(SourceResolver):
    def __init__(
        self, source_string: str, fallback_repo_id: str | None, file_name_map: Mapping[str, str] | None = None
    ):
        self.source = source_string
        self.fallback_repo_id = fallback_repo_id
        self.file_name_map = file_name_map

    def resolve(self, variant: ModelVariant, **kwargs) -> ArtifactMap:
        local_candidate = Path(self.source).expanduser()

        if local_candidate.is_dir():
            try:
                # Try strictly local first
                return LocalResolver(local_candidate, self.file_name_map).resolve(variant, **kwargs)
            except LoaderError as local_exc:
                if not self.fallback_repo_id:
                    raise LoaderError(f"Local resolution failed and no HF fallback configured: {local_exc}")
                logger.info(f"Local resolution incomplete, falling back to HF repo '{self.fallback_repo_id}'")

                # If local failed, fallback to HF entirely.
                # (Mixed local/HF merging is intentionally removed here for predictability.
                # Caching handles reuse automatically.)
                return HFResolver(self.fallback_repo_id, self.file_name_map).resolve(variant, **kwargs)

        # Treat source as HF repo
        return HFResolver(self.source, self.file_name_map).resolve(variant, **kwargs)


def resolve_variant_artifacts(
    source: str,
    variant: ModelVariant,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    fallback_hf_repo_id: str | None = None,
) -> ArtifactMap:
    """Main entrypoint for session factory to resolve files for a specific variant."""

    source = source.strip()
    if source.startswith("local:"):
        resolver = LocalResolver(Path(source[6:]).expanduser(), file_name_map)
    elif source.startswith("hf:"):
        resolver = HFResolver(source[3:], file_name_map)
    else:
        resolver = AutoResolver(source, fallback_hf_repo_id, file_name_map)

    return resolver.resolve(variant, revision=revision, cache_dir=cache_dir, allow_download=allow_download)
