"""Helpers for character -> copyright/IP mapping files."""

from __future__ import annotations

import ast
import csv
import json
import logging
from pathlib import Path

from vibe.hf_downloader import HFDownloadError, download_or_cached

logger = logging.getLogger(__name__)


DEFAULT_MAP_REPO = "deepghs/pixai-tagger-v0.9-onnx"
DEFAULT_MAP_FILE = "selected_tags.csv"


# region Load Map


def resolve_character_ip_mapping(
    manual_path: str | None = None,
    allow_download: bool | None = None,
    token: str | None = None,
    required: bool = True,
) -> dict[str, list[str]]:
    """
    Resolve mapping in priority order:
      1) explicit manual path
      2) community fallback file from HF
    """
    if manual_path:
        path = Path(manual_path)
        if not path.is_file():
            raise FileNotFoundError(f"Specified character IP mapping file does not exist: '{manual_path}'")
        return _load_mapping_file(path)

    fallback = download_or_cached(
        repo_id=DEFAULT_MAP_REPO,
        filename=DEFAULT_MAP_FILE,
        allow_download=allow_download,
        required=required,
        token=token,
    )

    if fallback is None:
        return {}
    return _load_mapping_file(fallback)


def _load_mapping_file(path: Path) -> dict[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_mapping_csv(path)
    if suffix in {".json", ".js"}:
        return _load_mapping_json(path)
    return {}


def _load_mapping_csv(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            ips_raw = row.get("ips", "[]")
            ips = _safe_parse_list(ips_raw)
            if not ips:
                continue

            out[name] = [str(x) for x in ips]

    return out


def _load_mapping_json(path: Path) -> dict[str, list[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load character mapping JSON %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        return {}

    # JSON format A:
    #   {"character_tag": ["ip1", "ip2"], ...}
    # or
    #   {"mapping": {"character_tag": ["ip1", "ip2"], ...}}
    candidate = raw.get("mapping") if isinstance(raw.get("mapping"), dict) else raw

    out: dict[str, list[str]] = {}
    for key, value in candidate.items():
        if not key or not isinstance(value, list) or not value:
            continue
        out[str(key)] = [str(x) for x in value]
    if out:
        return out

    # JSON format B:
    #   {
    #     "tag_map": {"char_tag": 123, ...},
    #     "ips_by_tag_id": {"123": ["ip1", "ip2"], ...}
    #   }
    tag_map = raw.get("tag_map")
    ips_by_tag_id = raw.get("ips_by_tag_id")
    if isinstance(tag_map, dict) and isinstance(ips_by_tag_id, dict):
        for tag_name, tag_id in tag_map.items():
            ips = ips_by_tag_id.get(str(tag_id), ips_by_tag_id.get(tag_id))
            if isinstance(ips, list) and ips:
                out[str(tag_name)] = [str(x) for x in ips]

    return out


# endregion Load Map


# region Map Utils


def _safe_parse_list(value: object) -> list[object]:
    if isinstance(value, list):
        return [item for item in value]
    if not isinstance(value, str):
        return []

    text = value.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Handle strings like [\"foo\", \"bar\"]
    if '\\"' in text:
        try:
            parsed = json.loads(text.replace('\\"', '"'))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return []


def apply_character_ip_mapping(
    character_tags: list[str],
    mapping: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return per-character mapped tags for predicted character labels."""
    if not character_tags or not mapping:
        return {}

    normalized = {k.replace(" ", "_"): v for k, v in mapping.items()}
    resolved: dict[str, list[str]] = {}

    for tag in character_tags:
        if tag in mapping:
            resolved[tag] = mapping[tag]
            continue

        normalized_tag = tag.replace(" ", "_")
        if normalized_tag in normalized:
            resolved[tag] = normalized[normalized_tag]

    return resolved


# endregion Map Utils
