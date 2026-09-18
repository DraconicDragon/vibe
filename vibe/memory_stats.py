"""Memory telemetry utilities for process and inference tracking."""

from __future__ import annotations

import atexit
import importlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)

_NVML_LOCK = threading.Lock()
_NVML_MODULE: Any | None = None
_NVML_INITIALIZED = False
_NVML_UNAVAILABLE = False

# region Memory Readers


def _empty_torch_cuda_stats() -> dict[str, int | None]:
    return {
        "allocated": None,
        "reserved": None,
        "max_allocated": None,
        "max_reserved": None,
    }


def _shutdown_nvml() -> None:
    global _NVML_INITIALIZED

    with _NVML_LOCK:
        if not _NVML_INITIALIZED or _NVML_MODULE is None:
            return
        try:
            _NVML_MODULE.nvmlShutdown()
        except Exception as exc:
            logger.debug("Failed to shutdown NVML during atexit cleanup: %s", exc)
        _NVML_INITIALIZED = False


def _get_nvml_module() -> Any | None:
    global _NVML_MODULE, _NVML_INITIALIZED, _NVML_UNAVAILABLE

    with _NVML_LOCK:
        if _NVML_UNAVAILABLE:
            return None

        if _NVML_MODULE is None:
            try:
                _NVML_MODULE = importlib.import_module("pynvml")
            except Exception:
                _NVML_UNAVAILABLE = True
                return None

        if not _NVML_INITIALIZED:
            try:
                _NVML_MODULE.nvmlInit()
            except Exception:
                _NVML_UNAVAILABLE = True
                return None
            _NVML_INITIALIZED = True
            atexit.register(_shutdown_nvml)

        return _NVML_MODULE


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _read_process_rss_bytes() -> int | None:
    # Linux fast path.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass

    # Fallback using resource where available.
    try:
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes, Linux reports KB.
        if os.name == "posix" and "darwin" in sys.platform:
            return ru_maxrss
        return ru_maxrss * 1024
    except (ImportError, AttributeError, OSError):
        return None


def _read_torch_cuda_stats() -> dict[str, int | None]:
    # Avoid importing torch as a side effect of telemetry snapshots.
    # If torch is not already loaded, leave torch-specific metrics unset.
    torch = sys.modules.get("torch")
    if torch is None:
        return _empty_torch_cuda_stats()

    try:
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if cuda is None or not callable(is_available) or not bool(is_available()):
            return _empty_torch_cuda_stats()

        return {
            "allocated": _safe_int(cuda.memory_allocated()),
            "reserved": _safe_int(cuda.memory_reserved()),
            "max_allocated": _safe_int(cuda.max_memory_allocated()),
            "max_reserved": _safe_int(cuda.max_memory_reserved()),
        }
    except (AttributeError, RuntimeError):
        return _empty_torch_cuda_stats()


def _read_nvml_process_bytes() -> int | None:
    # Optional cross-runtime GPU process memory via NVML.
    pynvml = _get_nvml_module()
    if pynvml is None:
        return None

    try:
        pid = os.getpid()
        used_total = 0
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle):
                if int(p.pid) == pid and p.usedGpuMemory is not None and p.usedGpuMemory >= 0:
                    used_total += int(p.usedGpuMemory)
        return used_total if used_total > 0 else None
    except Exception:
        return None


def _read_nvml_device_memory_bytes() -> tuple[int | None, int | None]:
    # Total GPU memory usage across devices (closer to what btop/nvidia-smi shows).
    pynvml = _get_nvml_module()
    if pynvml is None:
        return None, None

    try:
        used_total = 0
        total_total = 0
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_total += int(info.used)
            total_total += int(info.total)

        if total_total <= 0:
            return None, None
        return used_total, total_total
    except Exception:
        return None, None


# endregion Memory Readers

# region Data Data Models


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory snapshot."""

    timestamp: float
    process_rss_bytes: int | None
    torch_cuda_allocated_bytes: int | None
    torch_cuda_reserved_bytes: int | None
    torch_cuda_max_allocated_bytes: int | None
    torch_cuda_max_reserved_bytes: int | None
    gpu_process_used_bytes: int | None
    gpu_device_used_bytes: int | None
    gpu_device_total_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "process_rss_bytes": self.process_rss_bytes,
            "torch_cuda_allocated_bytes": self.torch_cuda_allocated_bytes,
            "torch_cuda_reserved_bytes": self.torch_cuda_reserved_bytes,
            "torch_cuda_max_allocated_bytes": self.torch_cuda_max_allocated_bytes,
            "torch_cuda_max_reserved_bytes": self.torch_cuda_max_reserved_bytes,
            "gpu_process_used_bytes": self.gpu_process_used_bytes,
            "gpu_device_used_bytes": self.gpu_device_used_bytes,
            "gpu_device_total_bytes": self.gpu_device_total_bytes,
        }


@dataclass(frozen=True)
class InferenceMemoryRecord:
    """Memory metrics for a single inference call."""

    index: int
    operation: str
    started_at: float
    finished_at: float
    elapsed_ms: float
    before: MemorySnapshot
    after: MemorySnapshot
    delta_process_rss_bytes: int | None
    delta_gpu_process_used_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation": self.operation,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "delta_process_rss_bytes": self.delta_process_rss_bytes,
            "delta_gpu_process_used_bytes": self.delta_gpu_process_used_bytes,
        }


@dataclass(frozen=True)
class MemoryTrackerStats:
    """Aggregated memory tracking state."""

    enabled: bool
    inference_calls: int
    last_record: InferenceMemoryRecord | None
    peak_process_rss_bytes: int | None
    peak_gpu_process_used_bytes: int | None
    max_inference_rss_delta_bytes: int | None
    max_inference_gpu_delta_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "inference_calls": self.inference_calls,
            "last_record": self.last_record.to_dict() if self.last_record else None,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "peak_gpu_process_used_bytes": self.peak_gpu_process_used_bytes,
            "max_inference_rss_delta_bytes": self.max_inference_rss_delta_bytes,
            "max_inference_gpu_delta_bytes": self.max_inference_gpu_delta_bytes,
        }


# endregion Data Models


# region MemoryTracker


class MemoryTracker:
    """Track per-call and peak memory usage for a session."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._calls = 0
        self._last_record: InferenceMemoryRecord | None = None
        self._peak_rss: int | None = None
        self._peak_gpu: int | None = None
        self._max_rss_delta: int | None = None
        self._max_gpu_delta: int | None = None

    def snapshot(self) -> MemorySnapshot:
        torch_stats = _read_torch_cuda_stats()
        gpu_device_used, gpu_device_total = _read_nvml_device_memory_bytes()
        return MemorySnapshot(
            timestamp=time.time(),
            process_rss_bytes=_read_process_rss_bytes(),
            torch_cuda_allocated_bytes=torch_stats["allocated"],
            torch_cuda_reserved_bytes=torch_stats["reserved"],
            torch_cuda_max_allocated_bytes=torch_stats["max_allocated"],
            torch_cuda_max_reserved_bytes=torch_stats["max_reserved"],
            gpu_process_used_bytes=_read_nvml_process_bytes(),
            gpu_device_used_bytes=gpu_device_used,
            gpu_device_total_bytes=gpu_device_total,
        )

    def observe(self, operation: str, before: MemorySnapshot, after: MemorySnapshot) -> InferenceMemoryRecord:
        self._calls += 1
        elapsed_ms = max(0.0, (after.timestamp - before.timestamp) * 1000.0)

        rss_delta = None
        if before.process_rss_bytes is not None and after.process_rss_bytes is not None:
            rss_delta = after.process_rss_bytes - before.process_rss_bytes

        gpu_delta = None
        if before.gpu_process_used_bytes is not None and after.gpu_process_used_bytes is not None:
            gpu_delta = after.gpu_process_used_bytes - before.gpu_process_used_bytes

        record = InferenceMemoryRecord(
            index=self._calls,
            operation=operation,
            started_at=before.timestamp,
            finished_at=after.timestamp,
            elapsed_ms=elapsed_ms,
            before=before,
            after=after,
            delta_process_rss_bytes=rss_delta,
            delta_gpu_process_used_bytes=gpu_delta,
        )
        self._last_record = record

        if after.process_rss_bytes is not None and (self._peak_rss is None or after.process_rss_bytes > self._peak_rss):
            self._peak_rss = after.process_rss_bytes

        if after.gpu_process_used_bytes is not None and (
            self._peak_gpu is None or after.gpu_process_used_bytes > self._peak_gpu
        ):
            self._peak_gpu = after.gpu_process_used_bytes

        if rss_delta is not None and (self._max_rss_delta is None or rss_delta > self._max_rss_delta):
            self._max_rss_delta = rss_delta

        if gpu_delta is not None and (self._max_gpu_delta is None or gpu_delta > self._max_gpu_delta):
            self._max_gpu_delta = gpu_delta

        return record

    def stats(self) -> MemoryTrackerStats:
        return MemoryTrackerStats(
            enabled=self.enabled,
            inference_calls=self._calls,
            last_record=self._last_record,
            peak_process_rss_bytes=self._peak_rss,
            peak_gpu_process_used_bytes=self._peak_gpu,
            max_inference_rss_delta_bytes=self._max_rss_delta,
            max_inference_gpu_delta_bytes=self._max_gpu_delta,
        )

    def reset(self) -> None:
        self._calls = 0
        self._last_record = None
        self._peak_rss = None
        self._peak_gpu = None
        self._max_rss_delta = None
        self._max_gpu_delta = None


# endregion MemoryTracker
