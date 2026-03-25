from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import autotagger
from autotagger.loader import ModelSource
from autotagger.session import build_session


class DummyONNXBackend:
    load_calls = 0
    close_calls = 0

    def __init__(self) -> None:
        self.providers = ["CPUExecutionProvider"]

    def load(
        self,
        weights_path: Path,
        providers: list[str] | None = None,
        input_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        del weights_path, providers, input_name, device
        type(self).load_calls += 1

    def run(self, array: np.ndarray) -> np.ndarray:
        batch = int(array.shape[0]) if hasattr(array, "shape") else 1
        # Matches test CSV shape: 4 tags [general, general, character, rating]
        base = np.array([[0.1, 0.2, 0.9, 0.3]], dtype=np.float32)
        return np.repeat(base, batch, axis=0)

    def close(self) -> None:
        type(self).close_calls += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run session lifecycle soak loops with memory telemetry logs.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--infer-per-session", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--gc-every", type=int, default=10)
    parser.add_argument(
        "--log-level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity. Use warning/error for quiet soak runs.",
    )
    parser.add_argument(
        "--skip-dummy",
        action="store_true",
        help="Skip dummy backend soak phase.",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Skip real ONNX backend soak phase.",
    )
    parser.add_argument(
        "--real-model-dir",
        type=str,
        default="",
        help="Path to local model directory for real ONNX phase (expects model.onnx + selected_tags.csv).",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    resolved_level = level_map.get(level.lower(), logging.DEBUG)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("autotagger.session").setLevel(resolved_level)


def _prepare_temp_source(root: Path) -> ModelSource:
    (root / "model.onnx").write_bytes(b"fake")
    (root / "selected_tags.csv").write_text(
        "name,category\n" "blue_hair,0\n" "cat_ears,0\n" "miku_hatsune,4\n" "safe,9\n",
        encoding="utf-8",
    )
    return ModelSource.local(root)


def _default_real_model_dirs() -> list[Path]:
    return [
        Path("models/wd-eva02-large-tagger-v3"),
        Path("models/caformer_b36.dbv4-full"),
    ]


def _resolve_real_model_dir(explicit: str) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            return candidate
        return None

    for candidate in _default_real_model_dirs():
        if candidate.is_dir():
            return candidate
    return None


def _select_real_providers() -> tuple[list[str], list[str]]:
    notes: list[str] = []
    try:
        from autotagger.backends.runtime.onnx import (
            prepare_onnxruntime_environment,
            resolve_onnx_provider_chain,
        )

        added_dirs = prepare_onnxruntime_environment()
        if added_dirs:
            notes.append(f"Prepared ONNX runtime library path with {len(added_dirs)} NVIDIA lib dir(s).")

        import onnxruntime as ort

        providers, _ = resolve_onnx_provider_chain(
            device="gpu",
            requested_providers=None,
            ort_module=ort,
        )
    except Exception as exc:
        notes.append(f"onnxruntime provider detection failed: {exc}")
        return ["CPUExecutionProvider"], notes

    if any(p != "CPUExecutionProvider" for p in providers):
        notes.append(f"Accelerator provider chain selected: {providers}")
    else:
        notes.append("No accelerator providers available; using CPUExecutionProvider.")
    return providers, notes


def _phase_summary(
    *,
    name: str,
    iterations: int,
    infer_per_session: int,
    elapsed_ms: list[float],
    rss_delta: list[int],
    gpu_delta: list[int],
    snapshot_rss: list[int],
    snapshot_gpu: list[int],
    snapshot_gpu_device_used: list[int],
    snapshot_gpu_device_total: list[int],
    snapshot_torch_reserved: list[int],
    notes: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase": name,
        "iterations": iterations,
        "infer_per_session": infer_per_session,
        "total_infer_calls": len(elapsed_ms),
        "avg_elapsed_ms": round(statistics.fmean(elapsed_ms), 3) if elapsed_ms else None,
        "p95_elapsed_ms": round(statistics.quantiles(elapsed_ms, n=20)[18], 3) if len(elapsed_ms) >= 20 else None,
        "max_elapsed_ms": round(max(elapsed_ms), 3) if elapsed_ms else None,
        "avg_host_rss_delta_bytes": int(statistics.fmean(rss_delta)) if rss_delta else None,
        "max_host_rss_delta_bytes": max(rss_delta) if rss_delta else None,
        "host_rss_baseline_min_bytes": min(snapshot_rss) if snapshot_rss else None,
        "host_rss_baseline_max_bytes": max(snapshot_rss) if snapshot_rss else None,
        "host_rss_baseline_drift_bytes": (max(snapshot_rss) - min(snapshot_rss)) if snapshot_rss else None,
        "avg_gpu_process_delta_bytes": int(statistics.fmean(gpu_delta)) if gpu_delta else None,
        "max_gpu_process_delta_bytes": max(gpu_delta) if gpu_delta else None,
        "gpu_process_baseline_min_bytes": min(snapshot_gpu) if snapshot_gpu else None,
        "gpu_process_baseline_max_bytes": max(snapshot_gpu) if snapshot_gpu else None,
        "gpu_process_baseline_drift_bytes": (max(snapshot_gpu) - min(snapshot_gpu)) if snapshot_gpu else None,
        "gpu_device_used_baseline_min_bytes": min(snapshot_gpu_device_used) if snapshot_gpu_device_used else None,
        "gpu_device_used_baseline_max_bytes": max(snapshot_gpu_device_used) if snapshot_gpu_device_used else None,
        "gpu_device_used_baseline_drift_bytes": (
            (max(snapshot_gpu_device_used) - min(snapshot_gpu_device_used)) if snapshot_gpu_device_used else None
        ),
        "gpu_device_total_min_bytes": min(snapshot_gpu_device_total) if snapshot_gpu_device_total else None,
        "gpu_device_total_max_bytes": max(snapshot_gpu_device_total) if snapshot_gpu_device_total else None,
        "torch_cuda_reserved_min_bytes": min(snapshot_torch_reserved) if snapshot_torch_reserved else None,
        "torch_cuda_reserved_max_bytes": max(snapshot_torch_reserved) if snapshot_torch_reserved else None,
        "torch_cuda_reserved_drift_bytes": (
            (max(snapshot_torch_reserved) - min(snapshot_torch_reserved)) if snapshot_torch_reserved else None
        ),
        "notes": notes,
    }
    if extra:
        summary["extra"] = extra
    return summary


def _run_phase(
    *,
    phase_name: str,
    source: ModelSource,
    iterations: int,
    infer_per_session: int,
    report_every: int,
    gc_every: int,
    onnx_providers: list[str] | None,
) -> tuple[dict[str, Any], int]:
    plugin_cls = autotagger.registry.get("wd-eva02-large")
    image = Image.new("RGB", (32, 32), (255, 0, 0))

    elapsed_ms: list[float] = []
    rss_delta: list[int] = []
    gpu_delta: list[int] = []
    snapshot_rss: list[int] = []
    snapshot_gpu: list[int] = []
    snapshot_gpu_device_used: list[int] = []
    snapshot_gpu_device_total: list[int] = []
    snapshot_torch_reserved: list[int] = []
    notes: list[str] = []

    for i in range(1, iterations + 1):
        session = build_session(
            plugin_cls=plugin_cls,
            source=source,
            backend="onnx",
            onnx_providers=onnx_providers,
            auto_download=False,
            memory_tracking=True,
        )

        for _ in range(infer_per_session):
            session.infer(image)

        stats = session.memory_stats()
        last_record = stats.get("last_record")
        if isinstance(last_record, dict):
            elapsed = last_record.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                elapsed_ms.append(float(elapsed))
            delta = last_record.get("delta_process_rss_bytes")
            if isinstance(delta, int):
                rss_delta.append(delta)
            gpu_mem_delta = last_record.get("delta_gpu_process_used_bytes")
            if isinstance(gpu_mem_delta, int):
                gpu_delta.append(gpu_mem_delta)

        snapshot = session.memory_snapshot()
        snapshot_rss_value = snapshot.get("process_rss_bytes")
        if isinstance(snapshot_rss_value, int):
            snapshot_rss.append(snapshot_rss_value)
        snapshot_gpu_value = snapshot.get("gpu_process_used_bytes")
        if isinstance(snapshot_gpu_value, int):
            snapshot_gpu.append(snapshot_gpu_value)
        snapshot_gpu_device_used_value = snapshot.get("gpu_device_used_bytes")
        if isinstance(snapshot_gpu_device_used_value, int):
            snapshot_gpu_device_used.append(snapshot_gpu_device_used_value)
        snapshot_gpu_device_total_value = snapshot.get("gpu_device_total_bytes")
        if isinstance(snapshot_gpu_device_total_value, int):
            snapshot_gpu_device_total.append(snapshot_gpu_device_total_value)
        snapshot_torch_reserved_value = snapshot.get("torch_cuda_reserved_bytes")
        if isinstance(snapshot_torch_reserved_value, int):
            snapshot_torch_reserved.append(snapshot_torch_reserved_value)

        if report_every > 0 and i % report_every == 0:
            print(f"[{phase_name}] iteration={i}")
            print(f"[{phase_name}] memory_stats=", json.dumps(stats, indent=2))
            print(f"[{phase_name}] memory_snapshot=", json.dumps(snapshot, indent=2))

        session.close()

        if i % gc_every == 0:
            gc.collect()

    if snapshot_rss:
        drift = max(snapshot_rss) - min(snapshot_rss)
        notes.append(f"Host RSS baseline drift across snapshots: {drift} bytes")
    if rss_delta:
        notes.append(f"Observed per-call host RSS deltas in range [{min(rss_delta)}, {max(rss_delta)}] bytes")
    if snapshot_gpu:
        gpu_drift = max(snapshot_gpu) - min(snapshot_gpu)
        notes.append(f"GPU process memory drift across snapshots: {gpu_drift} bytes")
    if snapshot_gpu_device_used:
        gpu_device_drift = max(snapshot_gpu_device_used) - min(snapshot_gpu_device_used)
        notes.append(f"GPU device-used drift across snapshots: {gpu_device_drift} bytes")
    if gpu_delta:
        notes.append(f"Observed per-call GPU deltas in range [{min(gpu_delta)}, {max(gpu_delta)}] bytes")
    if snapshot_torch_reserved:
        torch_reserved_drift = max(snapshot_torch_reserved) - min(snapshot_torch_reserved)
        notes.append(f"Torch CUDA reserved drift across snapshots: {torch_reserved_drift} bytes")
    if not snapshot_gpu and not snapshot_torch_reserved:
        notes.append("No GPU memory counters available in this phase (likely CPU-only runtime/path).")

    return (
        _phase_summary(
            name=phase_name,
            iterations=iterations,
            infer_per_session=infer_per_session,
            elapsed_ms=elapsed_ms,
            rss_delta=rss_delta,
            gpu_delta=gpu_delta,
            snapshot_rss=snapshot_rss,
            snapshot_gpu=snapshot_gpu,
            snapshot_gpu_device_used=snapshot_gpu_device_used,
            snapshot_gpu_device_total=snapshot_gpu_device_total,
            snapshot_torch_reserved=snapshot_torch_reserved,
            notes=notes,
        ),
        0,
    )


def _print_final_summary(summaries: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("SOAK SUMMARY")
    print("=" * 80)
    for item in summaries:
        print(json.dumps(item, indent=2))
        print("-" * 80)
    print("=" * 80)
    print("END SOAK SUMMARY")
    print("=" * 80)


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)

    if args.skip_dummy and args.skip_real:
        print("Nothing to run: both phases were skipped.")
        return 1

    summaries: list[dict[str, Any]] = []

    if not args.skip_dummy:
        import autotagger.backends.runtime.onnx as onnx_runtime

        original_onnx_backend = onnx_runtime.ONNXBackend
        onnx_runtime.ONNXBackend = DummyONNXBackend  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                source = _prepare_temp_source(Path(td))
                summary, _ = _run_phase(
                    phase_name="SOAK-DUMMY",
                    source=source,
                    iterations=args.iterations,
                    infer_per_session=args.infer_per_session,
                    report_every=args.report_every,
                    gc_every=args.gc_every,
                    onnx_providers=None,
                )
            summary["extra"] = {
                "backend_load_calls": DummyONNXBackend.load_calls,
                "backend_close_calls": DummyONNXBackend.close_calls,
                "load_close_balanced": DummyONNXBackend.load_calls == DummyONNXBackend.close_calls,
            }
            summaries.append(summary)
        finally:
            onnx_runtime.ONNXBackend = original_onnx_backend

    if not args.skip_real:
        notes: list[str] = []
        real_model_dir = _resolve_real_model_dir(args.real_model_dir)
        if real_model_dir is None:
            summaries.append(
                {
                    "phase": "SOAK-REAL",
                    "skipped": True,
                    "reason": "No real model directory found. Provide --real-model-dir.",
                }
            )
        else:
            providers, provider_notes = _select_real_providers()
            notes.extend(provider_notes)
            summary, _ = _run_phase(
                phase_name="SOAK-REAL",
                source=ModelSource.local(real_model_dir),
                iterations=args.iterations,
                infer_per_session=args.infer_per_session,
                report_every=args.report_every,
                gc_every=args.gc_every,
                onnx_providers=providers if providers else None,
            )
            existing_notes = summary.get("notes")
            if isinstance(existing_notes, list):
                summary["notes"] = existing_notes + notes
            else:
                summary["notes"] = notes
            summary["extra"] = {
                "model_dir": str(real_model_dir),
                "requested_providers": providers,
            }
            summaries.append(summary)

    _print_final_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
