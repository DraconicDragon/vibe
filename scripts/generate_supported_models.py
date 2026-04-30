from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import vibe

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_PATH = ROOT / "SUPPORTED_MODELS.md"
GITHUB_HF_BASE = "https://huggingface.co"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _escape_html(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _link(label: str, url: str) -> str:
    return f"[{_escape_markdown_cell(label)}]({url})"


def _code(text: str) -> str:
    escaped = text.replace("`", "\\`")
    return f"`{escaped}`"


def _format_backends(info: vibe.ModelPluginInfo) -> str:
    values = [backend.value for backend in info.supported_backends]
    return ", ".join(_code(value) for value in values) if values else "—"


def _format_required_files(info: vibe.ModelPluginInfo) -> str:
    if not info.required_files:
        return "—"

    repo = info.default_hf_repo
    parts: list[str] = []
    for spec in info.required_files:
        if repo:
            file_url = f"{GITHUB_HF_BASE}/{repo}/resolve/main/{spec.name}"
            file_label = _link(spec.name, file_url)
        else:
            file_label = _code(spec.name)

        if spec.backends:
            backends = "/".join(backend.value for backend in spec.backends)
            parts.append(f"{file_label} ({_code(backends)})")
        else:
            parts.append(file_label)
    return ", ".join(parts)


def _format_optional_text(value: str | None) -> str:
    if value is None or not value.strip():
        return "—"
    return _escape_markdown_cell(value.strip())


def _format_processors(info: vibe.ModelPluginInfo) -> str:
    if not info.supported_processors:
        return "—"
    return ", ".join(_code(processor) for processor in info.supported_processors)


def _format_aliases(info: vibe.ModelPluginInfo) -> str:
    if not info.aliases:
        return "—"
    return ", ".join(_code(alias) for alias in info.aliases)


def _family_label(plugin_cls: type[vibe.ModelPlugin]) -> str:
    module_name = plugin_cls.__module__.split(".")[-1]
    family_map = {
        "wd_tagger": "WD Tagger",
        "wdv4_animetimm": "AnimeTimm",
    }
    if module_name in family_map:
        return family_map[module_name]

    class_name = plugin_cls.__name__
    if class_name.endswith("Plugin"):
        class_name = class_name[:-6]
    return class_name


def _format_model_title(info: vibe.ModelPluginInfo) -> str:
    return f"<strong>{_escape_html(info.display_name)}</strong>"


def _format_model_section(info: vibe.ModelPluginInfo) -> str:
    title = _format_model_title(info)
    lines = [
        "<details>",
        f"<summary>{title}</summary>",
        "",
        f"- Model ID: {_code(info.model_id)}",
        f"- Aliases: {_format_aliases(info)}",
        f"- Source HF repo: {_link(info.default_hf_repo, f'{GITHUB_HF_BASE}/{info.default_hf_repo}') if info.default_hf_repo else '—'}",
        f"- Required files: {_format_required_files(info)}",
        f"- Backends: {_format_backends(info)}",
        f"- Output: {_code(info.output_type.value)}",
        f"- Result processors: {_format_processors(info)}",
        f"- Description: {_format_optional_text(info.description)}",
        "",
        "</details>",
    ]
    return "\n".join(lines)


def build_markdown() -> str:
    infos = sorted(vibe.describe_all(), key=lambda info: info.model_id)
    grouped: dict[str, list[vibe.ModelPluginInfo]] = defaultdict(list)

    for info in infos:
        plugin_cls = vibe.model_registry.get(info.model_id)
        grouped[_family_label(plugin_cls)].append(info)

    lines = [
        "# Supported Models",
        "",
        "Generated from `vibe.describe_all()` by `scripts/generate_supported_models.py`.",
        "",
        f"Total models: {len(infos)}.",
        "",
    ]

    for family_name in sorted(grouped.keys()):
        family_models = sorted(grouped[family_name], key=lambda info: info.display_name.lower() or info.model_id)
        lines.extend(
            [
                "<details>",
                f"<summary><strong>{_escape_html(family_name)}</strong> ({len(family_models)} models)</summary>",
                "",
            ]
        )

        for info in family_models:
            lines.append(_format_model_section(info))
            lines.append("")

        lines.append("</details>")
        lines.extend(
            [
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
