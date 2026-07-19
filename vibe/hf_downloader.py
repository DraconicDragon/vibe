"""Central HuggingFace downloader and download policy controls."""

from __future__ import annotations

import logging
from pathlib import Path

AUTO_DOWNLOAD_DEFAULT = True
logger = logging.getLogger(__name__)


# region Policy


class HFDownloadError(Exception):
    """Raised when a HuggingFace download/cached lookup cannot be satisfied."""


def _response_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _format_hf_access_error(repo_id: str, filename: str, exc: Exception) -> str:
    status_code = _response_status_code(exc)
    base = f"Failed to access '{filename}' in HuggingFace repo '{repo_id}'"
    if status_code in {401, 403}:
        return (
            f"{base} (HTTP {status_code}): unauthorized or forbidden. "
            "Check that your HuggingFace token is configured and that you have access to the repo."
        )
    if status_code == 404:
        return f"{base} (HTTP 404): repo or file not found."
    if status_code is not None:
        return f"{base} (HTTP {status_code})."
    return f"{base}."


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
    resolved, _reason = download_or_cached_with_reason(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        allow_download=allow_download,
        required=required,
    )
    return resolved


def download_or_cached_with_reason(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_download: bool | None = None,
    required: bool = True,
) -> tuple[Path | None, str | None]:
    """Resolve file from HF with optional reason string for unresolved optional files."""
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
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
    if isinstance(cached, str) and Path(cached).is_file():
        logger.debug("HF cache hit repo='%s' file='%s' -> %s", repo_id, filename, cached)
        return Path(cached), None

    auto_download_enabled = is_auto_download_enabled(allow_download)
    logger.debug(
        "HF cache miss repo='%s' file='%s' auto_download=%s required=%s",
        repo_id,
        filename,
        auto_download_enabled,
        required,
    )

    if not auto_download_enabled:
        reason = f"auto-download disabled and '{filename}' is not present in local cache"
        if required:
            raise HFDownloadError(f"Auto-download disabled and '{filename}' is not in cache for '{repo_id}'.")
        return None, reason

    try:
        logger.debug("Downloading HF file repo='%s' file='%s'", repo_id, filename)
        logger.info("Downloading '%s' from HuggingFace repo '%s'", filename, repo_id)
        resolved = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        logger.debug("Downloaded HF file repo='%s' file='%s'", repo_id, filename)
        return Path(resolved), None
    except EntryNotFoundError:
        reason = f"'{filename}' does not exist in repo '{repo_id}'"
        if required:
            raise HFDownloadError(f"Required file '{filename}' was not found in HF repo '{repo_id}'.") from None
        return None, reason
    except LocalEntryNotFoundError:
        reason = f"'{filename}' is unavailable in local cache for repo '{repo_id}'"
        if required:
            raise HFDownloadError(f"File '{filename}' not available in local cache for '{repo_id}'.") from None
        return None, reason
    except RepositoryNotFoundError as exc:
        reason = _format_hf_access_error(repo_id, filename, exc)
        if not required:
            return None, reason
        raise HFDownloadError(reason) from None
    except HfHubHTTPError as exc:
        reason = _format_hf_access_error(repo_id, filename, exc)
        if not required:
            return None, reason
        raise HFDownloadError(reason) from None


# endregion Resolve
