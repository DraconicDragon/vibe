"""
File loader — resolves model files from three sources:

  1. HuggingFace repo ID (auto-download via huggingface_hub, reuses HF cache)
  2. Local folder (user manually placed files; no HF involvement)
  3. Explicit HF cache path (user points directly to an existing HF snapshot)

The loader validates that required files are present before returning paths,
so plugins never have to deal with missing file errors at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autotagger.hf_downloader import HFDownloadError, download_or_cached

if TYPE_CHECKING:
    from autotagger.backends.base import Backend, FileSpec


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

    for spec in needed:
        try:
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
        except HFDownloadError as exc:
            raise LoaderError(str(exc)) from None

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

    for spec in needed:
        candidate = folder / spec.name
        if candidate.is_file():
            paths[spec.name] = candidate
        elif spec.required:
            # List what IS in the folder to help the user debug
            present = [f.name for f in folder.iterdir() if f.is_file()]
            raise LoaderError(
                f"Required file '{spec.name}' not found in {folder}.\n"
                f"Files present: {present}\n"
                f"Rename your file to match exactly, or check the plugin docs."
            )
        # else: optional, skip silently

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


# endregion Resolve Files

# region ModelSource


class ModelSource:
    """
    Describes where to load model files from.

    Use the class methods to construct:
        ModelSource.hf("SmilingWolf/wd-eva02-large-tagger-v3")
        ModelSource.local("/path/to/my/model/folder")
        ModelSource.hf_cache("/path/to/hf/snapshot")
        ModelSource.hf("some/repo", revision="v1.0")
    """

    _kind: str
    _path_or_id: str
    _revision: str | None
    _cache_dir: str | None
    _allow_download: bool | None

    def __init__(
        self,
        kind: str,
        path_or_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
    ) -> None:
        self._kind = kind
        self._path_or_id = path_or_id
        self._revision = revision
        self._cache_dir = cache_dir
        self._allow_download = allow_download

    @classmethod
    def hf(
        cls,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
    ) -> ModelSource:
        return cls(
            "hf",
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )

    @classmethod
    def local(cls, folder: str | Path) -> ModelSource:
        return cls("local", str(folder))

    @classmethod
    def hf_cache(cls, snapshot_path: str | Path) -> ModelSource:
        return cls("hf_cache", str(snapshot_path))

    def resolve(
        self,
        file_specs: list[FileSpec],
        backend: Backend,
    ) -> FileMap:
        """Resolve all needed files and return a FileMap."""
        if self._kind == "hf":
            return resolve_from_hf_repo(
                self._path_or_id,
                file_specs,
                backend,
                revision=self._revision,
                cache_dir=self._cache_dir,
                allow_download=self._allow_download,
            )
        elif self._kind in ("local", "hf_cache"):
            return resolve_from_local_folder(
                Path(self._path_or_id),
                file_specs,
                backend,
            )
        else:
            raise LoaderError(f"Unknown source kind: {self._kind!r}")

    def __repr__(self) -> str:
        if self._kind == "hf":
            rev = f"@{self._revision}" if self._revision else ""
            return f"ModelSource.hf({self._path_or_id!r}{rev})"
        return f"ModelSource.{self._kind}({self._path_or_id!r})"


# endregion ModelSource
