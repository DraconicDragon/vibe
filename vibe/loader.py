"""File loader and source string resolution for model assets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

from vibe.backends.base import ArtifactMap, ArtifactSpec, ModelVariant
from vibe.hf_downloader import HFDownloadError, download_or_cached_with_reason

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised when file resolution or validation fails."""


class SourceResolver(ABC):
    def __init__(self, file_name_map: Mapping[str, str] | None, source_map: Mapping[str, str] | None):
        self.file_name_map = file_name_map or {}
        self.source_map = source_map or {}

    def resolve(self, variant: ModelVariant, **kwargs) -> ArtifactMap:
        paths: dict[str, Path] = {}
        optional_missing: dict[str, str] = {}

        for spec in variant.artifacts:
            mapped_name = self.file_name_map.get(spec.id, spec.name)
            override_source = self.source_map.get(spec.id)

            # 1. Handle Explicit Artifact Override (via source_map)
            if override_source:
                path, reason = self._resolve_override(spec, override_source, mapped_name, **kwargs)
                if path:
                    paths[spec.id] = path
                elif not spec.required and reason:
                    optional_missing[spec.id] = reason
                else:
                    raise LoaderError(
                        f"Required artifact '{spec.id}' failed to load from override '{override_source}': {reason}"
                    )
                continue

            # 2. Delegate standard resolution to subclass
            path, reason = self._resolve_standard(spec, mapped_name, **kwargs)
            if path:
                paths[spec.id] = path
            elif not spec.required and reason:
                optional_missing[spec.id] = reason
            elif spec.required:
                raise LoaderError(reason or f"Required artifact '{spec.id}' could not be resolved.")

        return ArtifactMap(paths, optional_missing)

    @abstractmethod
    def _resolve_standard(self, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        pass

    def _resolve_override(
        self, spec: ArtifactSpec, override_source: str, mapped_name: str, **kwargs
    ) -> tuple[Path | None, str | None]:
        """Resolves a single artifact from an explicit override string."""
        override_source = override_source.strip()

        if override_source.startswith("local:"):
            candidate = Path(override_source[6:]).expanduser()
            if candidate.is_dir():
                candidate = candidate / mapped_name
            if candidate.is_file():
                return candidate, None
            return None, f"Local override file not found: {candidate}"

        if override_source.startswith("hf:"):
            repo_id = override_source[3:]
            return self._fetch_hf(repo_id, spec, mapped_name, **kwargs)

        # Unprefixed auto-mode override
        candidate = Path(override_source).expanduser()
        if candidate.is_dir():
            candidate = candidate / mapped_name
        if candidate.is_file():
            return candidate, None

        return None, f"Override source '{override_source}' could not be resolved locally."

    def _fetch_hf(self, repo_id: str, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        """Helper to fetch from Hugging Face cache/download."""
        hf_mapped_name = f"{spec.hf_subdir.rstrip('/')}/{mapped_name}" if spec.hf_subdir else mapped_name
        try:
            path, reason = download_or_cached_with_reason(
                repo_id=repo_id,
                filename=hf_mapped_name,
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                allow_download=kwargs.get("allow_download", True),
                required=spec.required,
            )
            return (Path(path) if path else None), reason
        except HFDownloadError as exc:
            return None, str(exc)


class LocalResolver(SourceResolver):
    def __init__(
        self, folder: Path, file_name_map: Mapping[str, str] | None = None, source_map: Mapping[str, str] | None = None
    ):
        super().__init__(file_name_map, source_map)
        self.folder = folder

    def _resolve_standard(self, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        # Strictly local. Ignore HF repo_ids entirely.
        if not self.folder.is_dir():
            return None, f"Local folder does not exist: {self.folder}"

        candidate = self.folder / mapped_name
        if candidate.is_file():
            return candidate, None
        return None, f"Missing in local folder: {candidate}"


class HFResolver(SourceResolver):
    def __init__(
        self,
        session_fallback_repo: str,
        file_name_map: Mapping[str, str] | None = None,
        source_map: Mapping[str, str] | None = None,
    ):
        super().__init__(file_name_map, source_map)
        self.session_fallback_repo = session_fallback_repo

    def _resolve_standard(self, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        # If the artifact defines its own repo (e.g. CLIP), use it. Otherwise use session repo.
        target_repo = spec.repo_id or self.session_fallback_repo

        if not target_repo:
            return None, "No repository ID defined for HF resolution."

        return self._fetch_hf(target_repo, spec, mapped_name, **kwargs)


class AutoResolver(SourceResolver):
    def __init__(
        self,
        folder: Path,
        session_fallback_repo: str | None,
        file_name_map: Mapping[str, str] | None = None,
        source_map: Mapping[str, str] | None = None,
    ):
        super().__init__(file_name_map, source_map)
        self.folder = folder
        self.session_fallback_repo = session_fallback_repo

    def _resolve_standard(self, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        # 1. Try local folder
        local_candidate = self.folder / mapped_name
        if local_candidate.is_file():
            return local_candidate, None

        # 2. Try HF fallback
        target_repo = spec.repo_id or self.session_fallback_repo
        if not target_repo:
            return None, "Missing locally, and no HF fallback repo available."

        return self._fetch_hf(target_repo, spec, mapped_name, **kwargs)


def resolve_variant_artifacts(
    source: str,
    variant: ModelVariant,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    fallback_hf_repo_id: str | None = None,
    source_map: Mapping[str, str] | None = None,
) -> ArtifactMap:
    """Main entrypoint for session factory to resolve files for a specific variant."""
    source = source.strip()

    if source.startswith("local:"):
        resolver = LocalResolver(Path(source[6:]).expanduser(), file_name_map, source_map)
    elif source.startswith("hf:"):
        resolver = HFResolver(source[3:], file_name_map, source_map)
    else:
        path_candidate = Path(source).expanduser()
        if path_candidate.is_dir():
            # It's a real local directory. If a file is missing, fallback to plugin default.
            resolver = AutoResolver(path_candidate, fallback_hf_repo_id, file_name_map, source_map)
        else:
            # It's not a directory, so treat the string as a Hugging Face repo ID.
            resolver = HFResolver(source, file_name_map, source_map)

    return resolver.resolve(variant, revision=revision, cache_dir=cache_dir, allow_download=allow_download)
