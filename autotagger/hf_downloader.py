"""Central HuggingFace downloader and download policy controls."""

from __future__ import annotations

from pathlib import Path

AUTO_DOWNLOAD_DEFAULT = True


# region Policy


class HFDownloadError(Exception):
    """Raised when a HuggingFace download/cached lookup cannot be satisfied."""


def set_auto_download_default(enabled: bool) -> None:
    """Set global default policy for auto-download behavior."""
    global AUTO_DOWNLOAD_DEFAULT
    AUTO_DOWNLOAD_DEFAULT = bool(enabled)


def get_auto_download_default() -> bool:
    """Get current global default policy for auto-download behavior."""
    return AUTO_DOWNLOAD_DEFAULT


def is_auto_download_enabled(allow_download: bool | None = None) -> bool:
    """
    Return effective download policy.

    `allow_download=None` means use the default global policy.
    """
    if allow_download is None:
        return AUTO_DOWNLOAD_DEFAULT
    return bool(allow_download)


# endregion Policy


# region Resolve


def download_or_cached(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    required: bool = True,
) -> Path | None:
    """
    Resolve file from HF cache, optionally downloading when permitted.

    Returns:
      - Path to the resolved file when available.
      - None when `required=False` and the file is unavailable.

    Raises HFDownloadError for required files that cannot be resolved.
    """
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        raise HFDownloadError(
            "huggingface_hub is required for HF file resolution. Install it with: pip install huggingface-hub"
        ) from exc

    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    if cached and Path(cached).is_file():
        return Path(cached)

    if not is_auto_download_enabled(allow_download):
        if required:
            raise HFDownloadError(f"Auto-download disabled and '{filename}' is not in cache for '{repo_id}'.")
        return None

    try:
        resolved = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        return Path(resolved)
    except EntryNotFoundError:
        if required:
            raise HFDownloadError(f"Required file '{filename}' was not found in HF repo '{repo_id}'.") from None
        return None
    except LocalEntryNotFoundError:
        if required:
            raise HFDownloadError(f"File '{filename}' not available in local cache for '{repo_id}'.") from None
        return None
    except RepositoryNotFoundError:
        raise HFDownloadError(f"HuggingFace repo '{repo_id}' was not found. Check repo ID and connectivity.") from None


# endregion Resolve
