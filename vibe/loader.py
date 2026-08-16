"""File loader and source string resolution for model assets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibe.backends.base import ArtifactMap, ArtifactSpec, Backend, FileRole, ModelVariant
from vibe.exceptions import HFDownloadError, LoaderError
from vibe.hf_downloader import download_or_cached_with_reason

logger = logging.getLogger(__name__)


def _is_local_source(source: str) -> bool:
    s = source.strip()
    if s.startswith("local:"):
        return True
    if s.startswith("hf:"):
        return False

    # Check for explicit local path syntax (Unix absolute/relative, home dir, Windows drive letter)
    if s.startswith(("/", "./", "../", "~")) or (len(s) >= 2 and s[1] == ":" and s[0].isalpha()):
        return True

    path = Path(s).expanduser()
    return path.is_dir() or path.is_absolute()


def _local_source_to_path(source: str) -> Path:
    source = source.strip()

    source = source.removeprefix("local:")

    return Path(source).expanduser()


def _hf_source_to_repo(source: str) -> str:
    return source.strip().removeprefix("hf:")


# region Data Classes


@dataclass(frozen=True)
class ArtifactAvailability:
    """Status of a single required or optional model artifact on disk/cache."""

    id: str
    name: str
    role: FileRole
    required: bool
    is_available: bool
    path: Path | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "required": self.required,
            "is_available": self.is_available,
            "path": str(self.path) if self.path else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VariantAvailability:
    """Status of a model variant's complete set of required artifacts."""

    variant_id: str | None
    backend: Backend
    is_available: bool
    artifacts: list[ArtifactAvailability]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "backend": self.backend.value,
            "is_available": self.is_available,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


@dataclass(frozen=True)
class ModelAvailability:
    """Overall availability summary for a model across its variants."""

    model_id: str
    is_available: bool
    variants: list[VariantAvailability]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_available": self.is_available,
            "variants": [v.to_dict() for v in self.variants],
        }


# endregion Data Classes


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
        self,
        spec: ArtifactSpec,
        override_source: str,
        mapped_name: str,
        **kwargs,
    ) -> tuple[Path | None, str | None]:

        if _is_local_source(override_source):
            candidate = _local_source_to_path(override_source)

            if candidate.is_dir():
                candidate = candidate / mapped_name

            if candidate.is_file():
                return candidate, None

            return None, f"Local override file not found: {candidate}"

        # Resolve overrides. Explicit overrides dictate their own layout, ignoring spec subfolders.
        repo_id, subfolder = parse_hf_source(override_source)
        return self._fetch_hf(
            repo_id,
            spec,
            mapped_name,
            override_subdir=subfolder,
            ignore_spec_subdir=True,
            **kwargs,
        )

    def _fetch_hf(
        self,
        repo_id: str,
        spec: ArtifactSpec,
        mapped_name: str,
        override_subdir: str | None = None,
        ignore_spec_subdir: bool = False,
        **kwargs,
    ) -> tuple[Path | None, str | None]:
        """Helper to fetch from Hugging Face cache/download."""
        if override_subdir is not None:
            subdir = override_subdir
        elif ignore_spec_subdir or not spec.hf_subdir:
            subdir = None
        else:
            subdir = spec.hf_subdir

        hf_mapped_name = f"{subdir.rstrip('/')}/{mapped_name}" if subdir else mapped_name
        try:
            path, reason = download_or_cached_with_reason(
                repo_id=repo_id,
                filename=hf_mapped_name,
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                allow_download=kwargs.get("allow_download", True),
                required=spec.required,
                token=kwargs.get("token"),
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

        # Check HF Repo-like nested structure first
        if spec.hf_subdir:
            nested_candidate = self.folder / spec.hf_subdir / mapped_name
            if nested_candidate.is_file():
                return nested_candidate, None

        # Flat structure
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
        # Parse the session's main source into repo_id and optional subfolder
        self.session_fallback_repo, self.session_fallback_subdir = parse_hf_source(session_fallback_repo)

    def _resolve_standard(self, spec: ArtifactSpec, mapped_name: str, **kwargs) -> tuple[Path | None, str | None]:
        # If the artifact defines its own repo (e.g. CLIP), use it. Otherwise use session repo.
        if spec.repo_id:
            target_repo = spec.repo_id
            target_subdir = spec.hf_subdir
        else:
            target_repo = self.session_fallback_repo
            # Fall back to spec's directory only if the session source did not specify a subfolder
            target_subdir = self.session_fallback_subdir or spec.hf_subdir

        if not target_repo:
            return None, "No repository ID defined for HF resolution."

        return self._fetch_hf(
            target_repo,
            spec,
            mapped_name,
            override_subdir=target_subdir,
            **kwargs,
        )


def resolve_variant_artifacts(
    source: str,
    variant: ModelVariant,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    source_map: Mapping[str, str] | None = None,
    token: str | None = None,
) -> ArtifactMap:
    """Main entrypoint for session factory to resolve files for a specific variant."""
    s = source.strip()

    if _is_local_source(s):
        folder = _local_source_to_path(s)
        resolver = LocalResolver(folder, file_name_map, source_map)
    else:
        repo = _hf_source_to_repo(s)
        resolver = HFResolver(repo, file_name_map, source_map)

    return resolver.resolve(
        variant,
        revision=revision,
        cache_dir=cache_dir,
        allow_download=allow_download,
        token=token,
    )


def inspect_variant_artifacts(
    source: str,
    variant: ModelVariant,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    file_name_map: Mapping[str, str] | None = None,
    source_map: Mapping[str, str] | None = None,
    token: str | None = None,
) -> list[ArtifactAvailability]:
    """Inspect local/cache availability for every artifact in a variant without downloading."""
    s = source.strip()

    if _is_local_source(s):
        folder = _local_source_to_path(s)
        resolver = LocalResolver(folder, file_name_map, source_map)
    else:
        repo = _hf_source_to_repo(s)
        resolver = HFResolver(repo, file_name_map, source_map)

    results: list[ArtifactAvailability] = []
    for spec in variant.artifacts:
        mapped_name = resolver.file_name_map.get(spec.id, spec.name)
        override_source = resolver.source_map.get(spec.id)

        if override_source:
            path, reason = resolver._resolve_override(
                spec,
                override_source,
                mapped_name,
                revision=revision,
                cache_dir=cache_dir,
                allow_download=False,
                token=token,
            )
        else:
            path, reason = resolver._resolve_standard(
                spec,
                mapped_name,
                revision=revision,
                cache_dir=cache_dir,
                allow_download=False,
                token=token,
            )

        is_avail = path is not None and path.is_file()
        results.append(
            ArtifactAvailability(
                id=spec.id,
                name=mapped_name,
                role=spec.role,
                required=spec.required,
                is_available=is_avail,
                path=path if is_avail else None,
                reason=None if is_avail else reason,
            )
        )

    return results


def parse_hf_source(source: str) -> tuple[str, str | None]:
    """
    Parses a Hugging Face source string into a valid (repo_id, subfolder) tuple.

    Examples:
        "username/repo-name" -> ("username/repo-name", None)
        "username/repo-name/models/clip" -> ("username/repo-name", "models/clip")
        "legacy-repo" -> ("legacy-repo", None)
    """
    source = source.strip()
    source = source.removeprefix("hf:")

    parts = source.split("/")
    if len(parts) > 2:
        # Standard user/repo/subfolder/... layout
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        return repo_id, subfolder

    # Either just "repo_id" or "username/repo_id" with no subfolder
    return source, None
