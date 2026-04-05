"""File loader and source string resolution for model assets."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from vibe.hf_downloader import HFDownloadError, download_or_cached

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vibe.backends.base import Backend, FileSpec


class LoaderError(Exception):
    """Raised when file resolution or validation fails."""


class FileMap:
    """
    A resolved set of file paths for a plugin.

    Access paths by filename:  file_map["model.onnx"]  → Path(...)
    """

    def __init__(self, paths: dict[str, Path]) -> None:
        self._paths = paths

    def __getitem__(self, name: str) -> Path:
        if name not in self._paths:
            raise KeyError(f"File '{name}' was not resolved. Available: {list(self._paths)}")
        return self._paths[name]

    def get(self, name: str) -> Path | None:
        return self._paths.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._paths

    def __repr__(self) -> str:
        return f"FileMap({self._paths})"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self._paths.items()}

    def as_path_dict(self) -> dict[str, Path]:
        """Return a copy of resolved paths keyed by filename."""
        return dict(self._paths)

    def values(self) -> list[Path]:
        return list(self._paths.values())


# region Resolve Files


def resolve_from_hf_repo(
    repo_id: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
) -> FileMap:
    """
    Download (or reuse cached) files from a HuggingFace repo.

    Only files that are needed for the given backend are fetched.
    Optional files that are absent from the repo are silently skipped.

    Args:
        repo_id:    HuggingFace repo ID, e.g. "SmilingWolf/wd-eva02-large-tagger-v3".
        file_specs: The plugin's required_files list.
        backend:    Which inference backend is being used.
        revision:   Git revision (branch/tag/commit). None = default branch.
        cache_dir:  Override the HF cache directory. None = HF default.
    """
    paths: dict[str, Path] = {}
    needed = [s for s in file_specs if s.needed_for(backend)]
    logger.debug(
        "Resolving model files from HF repo '%s' backend=%s files=%d",
        repo_id,
        backend.value,
        len(needed),
    )

    for spec in needed:
        try:
            logger.debug(
                "Resolving file from HF repo='%s' filename='%s' required=%s",
                repo_id,
                spec.name,
                spec.required,
            )
            local = download_or_cached(
                repo_id=repo_id,
                filename=spec.name,
                revision=revision,
                cache_dir=cache_dir,
                allow_download=allow_download,
                required=spec.required,
            )
            if local is not None:
                paths[spec.name] = Path(local)
                logger.debug("Resolved file '%s' -> %s", spec.name, local)
        except HFDownloadError as exc:
            raise LoaderError(str(exc)) from None
        except Exception as exc:
            # Keep loader API error surface consistent (LoaderError), including
            # hub-side validation errors such as invalid repo ID format.
            raise LoaderError(str(exc)) from None

    logger.debug("Resolved %d file(s) from HF repo '%s'", len(paths), repo_id)
    return FileMap(paths)


def resolve_from_local_folder(
    folder: Path,
    file_specs: list[FileSpec],
    backend: Backend,
) -> FileMap:
    """
    Resolve files from a local folder.

    The folder must contain files whose names match FileSpec.name exactly
    for all required files. Optional files that are absent are silently skipped.

    This path requires zero HuggingFace involvement — users just put files
    in a folder. The folder name doesn't matter here; callers can enforce a
    naming convention at the session/CLI level if desired.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise LoaderError(f"Local folder does not exist: {folder}")

    paths: dict[str, Path] = {}
    needed = [s for s in file_specs if s.needed_for(backend)]
    logger.debug(
        "Resolving model files from local folder '%s' backend=%s files=%d",
        folder,
        backend.value,
        len(needed),
    )

    for spec in needed:
        candidate = folder / spec.name
        if candidate.is_file():
            paths[spec.name] = candidate
            logger.debug("Resolved local file '%s' -> %s", spec.name, candidate)
        elif spec.required:
            # List what IS in the folder to help the user debug
            present = [f.name for f in folder.iterdir() if f.is_file()]
            raise LoaderError(
                f"Required file '{spec.name}' not found in {folder}.\n"
                f"Files present: {present}\n"
                f"Rename your file to match exactly, or check the plugin docs."
            )
        # else: optional, skip silently

    logger.debug("Resolved %d file(s) from local folder '%s'", len(paths), folder)
    return FileMap(paths)


def resolve_from_hf_cache_path(
    snapshot_path: Path,
    file_specs: list[FileSpec],
    backend: Backend,
) -> FileMap:
    """
    Resolve files from an already-downloaded HF snapshot directory.

    This is for users who point directly to a path inside their HF cache
    (e.g. ~/.cache/huggingface/hub/models--SmilingWolf--wd.../snapshots/abc123/).
    Behaves identically to resolve_from_local_folder — it's a separate function
    purely for clarity at the call site.
    """
    return resolve_from_local_folder(snapshot_path, file_specs, backend)


def resolve_from_source_string(
    source: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
) -> FileMap:
    """
    Resolve files from a user-facing source string.

    Supported source formats:
      - "local:/path/to/folder"  (strict local mode)
      - "hf:owner/repo"          (strict HuggingFace mode)
      - "hf_cache:/path"         (strict local HF snapshot mode)
      - unprefixed text           (auto mode: local folder if it exists, then HF repo)

    Prefix modes are strict and do not fall back to other source kinds.
    """
    source = source.strip()
    if not source:
        raise LoaderError("Source cannot be empty.")

    logger.debug("Resolving source '%s' for backend=%s", source, backend.value)

    if source.startswith("local:"):
        return _resolve_local_prefixed(source[6:], file_specs, backend)

    if source.startswith("hf:"):
        return _resolve_hf_prefixed(
            source[3:],
            file_specs,
            backend,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )

    if source.startswith("hf_cache:"):
        return _resolve_hf_cache_prefixed(source[9:], file_specs, backend)

    return _resolve_auto_source(
        source,
        file_specs,
        backend,
        revision=revision,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )


def _resolve_local_prefixed(
    raw_value: str,
    file_specs: list[FileSpec],
    backend: Backend,
) -> FileMap:
    value = raw_value.strip()
    if not value:
        raise LoaderError("Local source prefix requires a folder path: local:/path/to/folder")

    folder = Path(value).expanduser()
    try:
        return resolve_from_local_folder(folder, file_specs, backend)
    except LoaderError as exc:
        hint = ""
        if _looks_like_hf_repo_id(value):
            hint = " It looks like a HuggingFace repo ID; use 'hf:<owner/repo>' instead if that was intended."
        raise LoaderError(f"Requested local source via 'local:' but failed to resolve '{value}': {exc}.{hint}".rstrip())


def _resolve_hf_prefixed(
    raw_value: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None,
    cache_dir: str | None,
    allow_download: bool | None,
) -> FileMap:
    value = raw_value.strip()
    if not value:
        raise LoaderError("HF source prefix requires a repo ID: hf:owner/repo")

    try:
        return resolve_from_hf_repo(
            value,
            file_specs,
            backend,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
    except LoaderError as exc:
        hint = ""
        if _looks_like_local_folder(value):
            hint = " It looks like a local folder; use 'local:/path' instead if that was intended."
        raise LoaderError(f"Requested HF source via 'hf:' but failed to resolve '{value}': {exc}.{hint}".rstrip())


def _resolve_hf_cache_prefixed(
    raw_value: str,
    file_specs: list[FileSpec],
    backend: Backend,
) -> FileMap:
    value = raw_value.strip()
    if not value:
        raise LoaderError("HF cache source prefix requires a path: hf_cache:/path/to/snapshot")

    path = Path(value).expanduser()
    try:
        return resolve_from_hf_cache_path(path, file_specs, backend)
    except LoaderError as exc:
        hint = ""
        if _looks_like_local_folder(value):
            hint = " It looks like a local folder path; verify this HF snapshot folder contains the required files."
        elif _looks_like_hf_repo_id(value):
            hint = " It looks like a HuggingFace repo ID; use 'hf:<owner/repo>' instead if that was intended."
        raise LoaderError(
            f"Requested HF cache source via 'hf_cache:' but failed to resolve '{value}': {exc}.{hint}".rstrip()
        )


def _resolve_auto_source(
    source: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None,
    cache_dir: str | None,
    allow_download: bool | None,
) -> FileMap:
    local_candidate = Path(source).expanduser()
    local_error: str | None = None

    if local_candidate.exists():
        if local_candidate.is_dir():
            try:
                return resolve_from_local_folder(local_candidate, file_specs, backend)
            except LoaderError as exc:
                local_error = str(exc)
        else:
            local_error = f"Local path exists but is not a directory: {local_candidate}"

    hf_error: str | None = None
    try:
        return resolve_from_hf_repo(
            source,
            file_specs,
            backend,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
    except LoaderError as exc:
        hf_error = str(exc)

    parts = [
        f"Could not resolve source '{source}'.",
        "Auto mode tries local folder first (when it exists), then HuggingFace repo/cache/download.",
    ]
    if local_error is not None:
        parts.append(f"Local attempt failed: {local_error}")
    if hf_error is not None:
        parts.append(f"HF attempt failed: {hf_error}")
    raise LoaderError(" ".join(parts))


def _looks_like_hf_repo_id(value: str) -> bool:
    return bool(re.match(r"^[^/\s]+/[^/\s]+$", value.strip()))


def _looks_like_local_folder(value: str) -> bool:
    p = Path(value).expanduser()
    if p.is_dir():
        return True

    raw = value.strip()
    return raw.startswith(("./", "../", "/", "~/"))


# endregion Resolve Files
