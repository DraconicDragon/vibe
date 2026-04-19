"""File loader and source string resolution for model assets."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from vibe.hf_downloader import HFDownloadError, download_or_cached_with_reason


def download_or_cached(*args, **kwargs):
    """Compatibility shim for tests/extensions patching vibe.loader.download_or_cached."""
    return download_or_cached_with_reason(*args, **kwargs)


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

    def __init__(
        self,
        paths: dict[str, Path],
        *,
        optional_missing_reasons: Mapping[str, str] | None = None,
    ) -> None:
        self._paths = paths
        self._optional_missing_reasons = dict(optional_missing_reasons or {})

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

    def optional_missing_reasons(self) -> dict[str, str]:
        """Return optional file resolution reasons keyed by plugin-declared filename."""
        return dict(self._optional_missing_reasons)

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
    file_name_map: Mapping[str, str] | None = None,
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
        file_name_map:
                    Optional mapping from plugin-declared filename to repo
                    filename override. Useful when a repo renamed files.
    """
    normalized_file_name_map = _normalize_file_name_map(file_name_map, file_specs)
    paths: dict[str, Path] = {}
    optional_missing: dict[str, str] = {}
    needed = [s for s in file_specs if s.needed_for(backend)]
    logger.debug(
        "Resolving model files from HF repo '%s' backend=%s files=%d",
        repo_id,
        backend.value,
        len(needed),
    )

    for spec in needed:
        mapped_name = normalized_file_name_map.get(spec.name, spec.name)
        try:
            logger.debug(
                "Resolving file from HF repo='%s' spec='%s' mapped='%s' required=%s",
                repo_id,
                spec.name,
                mapped_name,
                spec.required,
            )
            download_result = download_or_cached(
                repo_id=repo_id,
                filename=mapped_name,
                revision=revision,
                cache_dir=cache_dir,
                allow_download=allow_download,
                required=spec.required,
            )
            local: Path | None
            optional_reason: str | None
            if isinstance(download_result, tuple):
                local, optional_reason = download_result
            else:
                local = download_result
                optional_reason = None
            if local is not None:
                paths[spec.name] = Path(local)
                logger.debug("Resolved HF file spec='%s' mapped='%s' -> %s", spec.name, mapped_name, local)
            elif not spec.required and optional_reason:
                optional_missing[spec.name] = optional_reason
        except HFDownloadError as exc:
            raise LoaderError(str(exc)) from None
        except Exception as exc:
            # Keep loader API error surface consistent (LoaderError), including
            # hub-side validation errors such as invalid repo ID format.
            raise LoaderError(str(exc)) from None

    logger.debug("Resolved %d file(s) from HF repo '%s'", len(paths), repo_id)
    return FileMap(paths, optional_missing_reasons=optional_missing)


def resolve_from_local_folder(
    folder: Path,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    file_name_map: Mapping[str, str] | None = None,
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

    normalized_file_name_map = _normalize_file_name_map(file_name_map, file_specs)
    paths: dict[str, Path] = {}
    optional_missing: dict[str, str] = {}
    needed = [s for s in file_specs if s.needed_for(backend)]
    logger.debug(
        "Resolving model files from local folder '%s' backend=%s files=%d",
        folder,
        backend.value,
        len(needed),
    )

    for spec in needed:
        mapped_name = normalized_file_name_map.get(spec.name, spec.name)
        candidate = folder / mapped_name
        if candidate.is_file():
            paths[spec.name] = candidate
            logger.debug(
                "Resolved local file spec='%s' mapped='%s' -> %s",
                spec.name,
                mapped_name,
                candidate,
            )
        elif spec.required:
            # List what IS in the folder to help the user debug
            present = [f.name for f in folder.iterdir() if f.is_file()]
            map_hint = ""
            if mapped_name != spec.name:
                map_hint = f" (mapped from '{spec.name}' to '{mapped_name}')"
            raise LoaderError(
                f"Required file '{mapped_name}' not found in {folder}{map_hint}.\n"
                f"Files present: {present}\n"
                f"Rename your file to match exactly, or check the plugin docs."
            )
        else:
            optional_missing[spec.name] = f"'{mapped_name}' not found in local folder '{folder}'"

    logger.debug("Resolved %d file(s) from local folder '%s'", len(paths), folder)
    return FileMap(paths, optional_missing_reasons=optional_missing)


def resolve_from_source_string(
    source: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    fallback_hf_repo_id: str | None = None,
) -> FileMap:
    """
    Resolve files from a user-facing source string.

    Supported source formats:
      - "local:/path/to/folder"  (strict local mode)
      - "hf:owner/repo"          (strict HuggingFace mode)
      - unprefixed text           (auto mode: local folder if it exists, then HF repo)

    Prefix modes are strict and do not fall back to other source kinds.
    """
    source = source.strip()
    if not source:
        raise LoaderError("Source cannot be empty.")

    logger.debug("Resolving source '%s' for backend=%s", source, backend.value)

    if source.startswith("local:"):
        return _resolve_local_prefixed(source[6:], file_specs, backend, file_name_map=file_name_map)

    if source.startswith("hf:"):
        return _resolve_hf_prefixed(
            source[3:],
            file_specs,
            backend,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
            file_name_map=file_name_map,
        )

    return _resolve_auto_source(
        source,
        file_specs,
        backend,
        revision=revision,
        cache_dir=cache_dir,
        allow_download=allow_download,
        file_name_map=file_name_map,
        fallback_hf_repo_id=fallback_hf_repo_id,
    )


def _resolve_local_prefixed(
    raw_value: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    file_name_map: Mapping[str, str] | None,
) -> FileMap:
    value = raw_value.strip()
    if not value:
        raise LoaderError("Local source prefix requires a folder path: local:/path/to/folder")

    folder = Path(value).expanduser()
    try:
        return resolve_from_local_folder(folder, file_specs, backend, file_name_map=file_name_map)
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
    file_name_map: Mapping[str, str] | None,
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
            file_name_map=file_name_map,
        )
    except LoaderError as exc:
        hint = ""
        if _looks_like_local_folder(value):
            hint = " It looks like a local folder; use 'local:/path' instead if that was intended."
        raise LoaderError(f"Requested HF source via 'hf:' but failed to resolve '{value}': {exc}.{hint}".rstrip())


def _resolve_auto_source(
    source: str,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None,
    cache_dir: str | None,
    allow_download: bool | None,
    file_name_map: Mapping[str, str] | None,
    fallback_hf_repo_id: str | None,
) -> FileMap:
    local_candidate = Path(source).expanduser()
    local_error: str | None = None

    if local_candidate.exists():
        if local_candidate.is_dir():
            try:
                return _resolve_local_then_hf_missing(
                    local_candidate,
                    file_specs,
                    backend,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_download=allow_download,
                    file_name_map=file_name_map,
                    fallback_hf_repo_id=fallback_hf_repo_id,
                )
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
            file_name_map=file_name_map,
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


def _resolve_local_then_hf_missing(
    folder: Path,
    file_specs: list[FileSpec],
    backend: Backend,
    *,
    revision: str | None,
    cache_dir: str | None,
    allow_download: bool | None,
    file_name_map: Mapping[str, str] | None,
    fallback_hf_repo_id: str | None,
) -> FileMap:
    normalized_file_name_map = _normalize_file_name_map(file_name_map, file_specs)
    needed = [s for s in file_specs if s.needed_for(backend)]

    paths: dict[str, Path] = {}
    optional_missing: dict[str, str] = {}
    missing_specs: list[FileSpec] = []

    for spec in needed:
        mapped_name = normalized_file_name_map.get(spec.name, spec.name)
        candidate = folder / mapped_name
        if candidate.is_file():
            paths[spec.name] = candidate
            continue

        if spec.required:
            missing_specs.append(spec)
        else:
            optional_missing[spec.name] = f"'{mapped_name}' not found in local folder '{folder}'"
            missing_specs.append(spec)

    if not missing_specs:
        return FileMap(paths, optional_missing_reasons=optional_missing)

    if not fallback_hf_repo_id:
        required_mapped = [
            normalized_file_name_map.get(spec.name, spec.name) for spec in missing_specs if spec.required
        ]
        if required_mapped:
            present = [f.name for f in folder.iterdir() if f.is_file()]
            raise LoaderError(
                f"Missing required file(s) in local folder '{folder}': {required_mapped}. "
                f"No fallback HF repo configured. Files present: {present}"
            )
        return FileMap(paths, optional_missing_reasons=optional_missing)

    for spec in missing_specs:
        mapped_name = normalized_file_name_map.get(spec.name, spec.name)
        try:
            local, reason = download_or_cached_with_reason(
                repo_id=fallback_hf_repo_id,
                filename=mapped_name,
                revision=revision,
                cache_dir=cache_dir,
                allow_download=allow_download,
                required=spec.required,
            )
            if local is not None:
                # TODO: Revisit whether local write-back should be optional/configurable.
                local_materialized = _materialize_downloaded_file_to_local_folder(
                    source_path=Path(local),
                    destination_folder=folder,
                    destination_name=mapped_name,
                )
                paths[spec.name] = local_materialized
                optional_missing.pop(spec.name, None)
                continue

            if not spec.required:
                local_reason = optional_missing.get(spec.name, "local file missing")
                hf_reason = reason or "HF fallback did not provide a reason"
                optional_missing[spec.name] = (
                    f"{local_reason}; HF fallback '{fallback_hf_repo_id}' could not resolve '{mapped_name}': {hf_reason}"
                )
        except HFDownloadError as exc:
            if spec.required:
                raise LoaderError(
                    f"Required file '{mapped_name}' not found in local folder '{folder}', "
                    f"and HF fallback '{fallback_hf_repo_id}' failed: {exc}"
                ) from None

            local_reason = optional_missing.get(spec.name, "local file missing")
            optional_missing[spec.name] = (
                f"{local_reason}; HF fallback '{fallback_hf_repo_id}' failed for '{mapped_name}': {exc}"
            )

    return FileMap(paths, optional_missing_reasons=optional_missing)


def _materialize_downloaded_file_to_local_folder(
    *,
    source_path: Path,
    destination_folder: Path,
    destination_name: str,
) -> Path:
    destination_path = destination_folder / destination_name
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() == destination_path.resolve():
        return destination_path

    shutil.copy2(source_path, destination_path)
    return destination_path


def _looks_like_hf_repo_id(value: str) -> bool:
    return bool(re.match(r"^[^/\s]+/[^/\s]+$", value.strip()))


def _looks_like_local_folder(value: str) -> bool:
    p = Path(value).expanduser()
    if p.is_dir():
        return True

    raw = value.strip()
    return raw.startswith(("./", "../", "/", "~/"))


def _normalize_file_name_map(
    file_name_map: Mapping[str, str] | None,
    file_specs: list[FileSpec],
) -> dict[str, str]:
    if not file_name_map:
        return {}

    allowed_names = {spec.name for spec in file_specs}
    normalized: dict[str, str] = {}
    for original_name, mapped_name in file_name_map.items():
        key = str(original_name).strip()
        value = str(mapped_name).strip()
        if not key:
            raise LoaderError("file_name_map has an empty key; expected original plugin file names.")
        if not value:
            raise LoaderError(f"file_name_map entry for '{key}' has an empty mapped filename.")
        if key not in allowed_names:
            raise LoaderError(
                f"file_name_map contains unknown key '{key}'. Known plugin files: {sorted(allowed_names)}"
            )
        normalized[key] = value

    return normalized


# endregion Resolve Files
