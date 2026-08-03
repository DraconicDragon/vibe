"""
ONNX Runtime inference backend.

Wraps an onnxruntime.InferenceSession and provides a uniform
.run(array) → ndarray interface.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from vibe import Backend
from vibe.backends.base import ExecutionRequest
from vibe.devices import normalize_device_string
from vibe.precision import PrecisionPolicy

logger = logging.getLogger(__name__)

# Auto-priority excludes TensorRT on purpose to keep behavior predictable.
ONNX_AUTO_PROVIDER_PRIORITY: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    # todo: add intel GPU EP, onednn or something?
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)

_GPU_CLASS_PROVIDERS: frozenset[str] = frozenset(
    {
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
    }
)

_ENV_PROVIDER_SINGLE = "VIBE_ONNX_PROVIDER"
_ENV_PROVIDER_LIST = "VIBE_ONNX_PROVIDERS"


# region Provider Setup


def _onnx_type_to_precision(type_name: str | None) -> str | None:
    if not type_name:
        return None
    value = str(type_name).lower()
    if "float16" in value:
        return "fp16"
    if "bfloat16" in value:
        return "bf16"
    if "float" in value:
        return "fp32"
    if "int8" in value:
        return "int8"
    return None


def _normalize_provider_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _providers_from_env() -> list[str] | None:
    raw_list = os.getenv(_ENV_PROVIDER_LIST, "")
    if raw_list.strip():
        parsed = _normalize_provider_list(raw_list.split(","))
        if parsed:
            return parsed

    raw_single = os.getenv(_ENV_PROVIDER_SINGLE, "")
    if raw_single.strip():
        parsed = _normalize_provider_list([raw_single])
        if parsed:
            return parsed

    return None


def _device_prefers_accelerator(device: str) -> tuple[bool, int]:
    value = normalize_device_string(device, backend=Backend.ONNX)
    if not value or value == "auto":
        return True, 0
    if value == "cpu":
        return False, 0
    if value in {"gpu", "cuda", "rocm", "dml"}:
        return True, 0

    for prefix in ("gpu:", "cuda:", "rocm:", "dml:"):
        if value.startswith(prefix):
            try:
                return True, max(0, int(value.split(":", 1)[1]))
            except ValueError:
                return True, 0

    return value != "cpu", 0


def _available_onnx_providers(ort_module: Any) -> list[str]:
    get_available = getattr(ort_module, "get_available_providers", None)
    if callable(get_available):
        return _normalize_provider_list([str(p) for p in get_available()])
    return []


def resolve_onnx_provider_chain(
    *,
    device: str,
    requested_providers: list[str] | None,
    ort_module: Any,
) -> tuple[list[str], list[dict[str, Any]] | None]:
    """Resolve providers and provider options from request/env/device state."""
    available = _available_onnx_providers(ort_module)
    available_set = set(available)

    explicit = requested_providers
    if explicit is None:
        explicit = _providers_from_env()

    wants_accelerator, device_id = _device_prefers_accelerator(device)

    if explicit is not None:
        providers = _normalize_provider_list([str(p) for p in explicit])
        if available:
            providers = [p for p in providers if p in available_set]
        if not providers and "CPUExecutionProvider" in available_set:
            providers = ["CPUExecutionProvider"]
    else:
        if wants_accelerator:
            preferred = [p for p in ONNX_AUTO_PROVIDER_PRIORITY if p != "CPUExecutionProvider"]
            if available:
                providers = [p for p in preferred if p in available_set]
            else:
                providers = preferred

            if available:
                if "CPUExecutionProvider" in available_set and "CPUExecutionProvider" not in providers:
                    providers.append("CPUExecutionProvider")
            elif "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
        else:
            if available:
                providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available_set else []
            else:
                providers = ["CPUExecutionProvider"]

    if not providers:
        providers = ["CPUExecutionProvider"]

    provider_options: list[dict[str, Any]] = []
    for provider in providers:
        if provider in _GPU_CLASS_PROVIDERS:
            provider_options.append({"device_id": str(device_id)})
        else:
            provider_options.append({})

    has_non_empty_options = any(bool(options) for options in provider_options)
    logger.debug(
        "ONNX providers resolved device=%s providers=%s",
        device,
        providers,
    )
    logger.debug(
        "ONNX provider resolution details available=%s explicit=%s provider_options=%s",
        available,
        explicit,
        provider_options if has_non_empty_options else None,
    )
    return providers, provider_options if has_non_empty_options else None


def _wants_accelerator(device: str) -> bool:
    try:
        value = normalize_device_string(device, backend=Backend.ONNX)
    except ValueError:
        return False
    return value not in {"cpu", "auto"}


def _requested_onnx_provider_name(device: str) -> str | None:
    try:
        value = normalize_device_string(device, backend=Backend.ONNX)
    except ValueError:
        return None

    if value in {"gpu", "cuda"}:
        return "CUDAExecutionProvider"
    if value.startswith(("gpu:", "cuda:")):
        return "CUDAExecutionProvider"
    if value.startswith("rocm"):
        return "ROCMExecutionProvider"
    if value.startswith("dml"):
        return "DmlExecutionProvider"
    if value == "mps":
        return "CoreMLExecutionProvider"
    return None


def _has_accelerator_provider(providers: list[str]) -> bool:
    accelerator_eps = _GPU_CLASS_PROVIDERS | {"CoreMLExecutionProvider", "DirectMLExecutionProvider"}
    return any(p in accelerator_eps for p in providers)


def _fallback_warning_message(device: str, resolved_providers: list[str], session_provider: str) -> str:
    requested_provider = _requested_onnx_provider_name(device)
    if requested_provider is None and resolved_providers:
        requested_provider = resolved_providers[0]
    requested_label = requested_provider or device.strip() or "auto"
    return (
        f"ONNX backend fell back to {session_provider} after ORT could not load the requested "
        f"provider {requested_label}."
    )


def _iter_candidate_nvidia_lib_dirs() -> list[Path]:
    # Common pip locations for NVIDIA runtime wheels used by torch/onnxruntime.
    candidates: list[Path] = []
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for base in [Path(sys.prefix), Path(sys.base_prefix)]:
        candidates.append(base / "lib" / major_minor / "site-packages" / "nvidia")
    return candidates


def _configure_linux_cuda_library_path() -> list[str]:
    """Append NVIDIA pip wheel lib directories to LD_LIBRARY_PATH on Linux."""
    if sys.platform != "linux":
        return []

    existing_parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    existing_set = set(existing_parts)
    discovered: list[str] = []

    for nvidia_root in _iter_candidate_nvidia_lib_dirs():
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if not lib_dir.is_dir():
                continue
            lib_str = str(lib_dir)
            if lib_str in existing_set:
                continue
            discovered.append(lib_str)
            existing_set.add(lib_str)

    if discovered:
        os.environ["LD_LIBRARY_PATH"] = ":".join(discovered + existing_parts)

    return discovered


_PRELOADED_NVIDIA_LIBS: set[str] = set()


def _candidate_cuda_library_paths(lib_dirs: list[str]) -> list[str]:
    patterns = [
        "libcudart.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcudnn.so*",
        "libcurand.so*",
        "libcufft.so*",
    ]
    paths: list[str] = []
    for lib_dir in lib_dirs:
        base = Path(lib_dir)
        if not base.is_dir():
            continue
        for pattern in patterns:
            for match in sorted(base.glob(pattern)):
                if match.is_file():
                    paths.append(str(match))
    return paths


def _preload_linux_nvidia_libraries(lib_dirs: list[str]) -> list[str]:
    if sys.platform != "linux":
        return []

    loaded: list[str] = []
    for lib_path in _candidate_cuda_library_paths(lib_dirs):
        if lib_path in _PRELOADED_NVIDIA_LIBS:
            continue
        try:
            # RTLD_GLOBAL helps subsequent provider library symbol resolution.
            ctypes.CDLL(lib_path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            _PRELOADED_NVIDIA_LIBS.add(lib_path)
            loaded.append(lib_path)
        except OSError:
            continue
    return loaded


def prepare_onnxruntime_environment() -> list[str]:
    """Prepare process environment for onnxruntime import on Linux."""
    discovered = _configure_linux_cuda_library_path()
    _preload_linux_nvidia_libraries(discovered)
    return discovered


# endregion Provider Setup


# Best-effort early bootstrap so any later onnxruntime imports in this process
# see the NVIDIA wheel runtime directories.
prepare_onnxruntime_environment()

# region ONNXBackend


class ONNXBackend:
    """
    Loads and runs an ONNX model via onnxruntime.

        Provider priority (auto-detected unless overridden):
            CUDAExecutionProvider → ROCMExecutionProvider → DmlExecutionProvider
            → CoreMLExecutionProvider → CPUExecutionProvider
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._input_name: str = ""
        self._output_names: list[str] | None = None
        self._providers: list[str] = []
        self._requested_providers: list[str] = []
        self._provider_options: list[dict[str, Any]] = []
        self._requested_precision: str = "auto"
        self._run_lock = threading.RLock()

    def load(
        self,
        model_path: Path,
        request: ExecutionRequest,
    ) -> None:
        """Load a plugin-selected ONNX graph for execution."""
        started_at = time.perf_counter()
        logger.debug("Loading ONNX model from %s", model_path)
        prepare_onnxruntime_environment()

        try:
            import onnxruntime as ort  # ty:ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to use the onnx backend. "
                "Install it with: pip install onnxruntime  "
                "(or onnxruntime-gpu for CUDA support)"
            ) from exc

        resolved_providers, resolved_provider_options = resolve_onnx_provider_chain(
            device=request.device,
            requested_providers=list(request.onnx_providers) if request.onnx_providers is not None else None,
            ort_module=ort,
        )

        self._requested_providers = list(resolved_providers)
        self._providers = resolved_providers
        self._provider_options = resolved_provider_options or []
        self._session = ort.InferenceSession(
            str(model_path),
            providers=resolved_providers,
            provider_options=resolved_provider_options,
        )
        load_seconds = time.perf_counter() - started_at
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_name = inputs[0].name
        input_meta = inputs[0]

        # Find the most appropriate output tensor if multiple exist
        self._output_names = None
        output_meta = outputs[0] if outputs else None
        if outputs:
            out_names = [o.name for o in outputs]
            target_name = out_names[0]
            if len(out_names) > 1:
                # Prioritize prediction/logits over embeddings
                for pref in (
                    "prediction",
                    "predictions",
                    "probs",
                    "probabilities",
                ):
                    if pref in out_names:
                        target_name = pref
                        break
            self._output_names = [target_name]
            output_meta = next((o for o in outputs if o.name == target_name), outputs[0])

        session_providers = _normalize_provider_list([str(provider) for provider in self._session.get_providers()])
        self._providers = session_providers
        primary_provider = self._providers[0] if self._providers else "CPUExecutionProvider"

        requested_provider = _requested_onnx_provider_name(request.device)
        requested_has_accelerator = _has_accelerator_provider(self._requested_providers)
        session_has_accelerator = _has_accelerator_provider(session_providers)

        if requested_has_accelerator and not session_has_accelerator and _wants_accelerator(request.device):
            raise RuntimeError(
                f"ONNX provider request '{request.device}' ({requested_provider or 'requested provider'}) could not be satisfied; "
                f"session loaded only with {session_providers}."
            )

        if requested_has_accelerator and not session_has_accelerator and request.device.strip().lower() == "auto":
            fallback_message = _fallback_warning_message(request.device, self._requested_providers, primary_provider)
            logger.warning(fallback_message)
            import warnings

            warnings.warn(fallback_message, RuntimeWarning, stacklevel=2)

        # precision setting not particularly useful for ONNX
        compute_prec = request.precision.compute
        if compute_prec in (PrecisionPolicy.FP16, PrecisionPolicy.BF16, PrecisionPolicy.FP32):
            logger.warning(
                "ONNX precision request '%s' is advisory only; most precision behavior is defined by model graph "
                "and execution provider kernels.",
                compute_prec.value,
            )

        logger.info("ONNX model loaded in %.2fs | session EP=%s", load_seconds, primary_provider)
        input_precision = _onnx_type_to_precision(getattr(input_meta, "type", None))
        output_precision = _onnx_type_to_precision(getattr(output_meta, "type", None))
        model_precision = input_precision or output_precision or "unknown"
        if input_precision and output_precision and input_precision != output_precision:
            model_precision = f"mixed(input={input_precision}, output={output_precision})"

        logger.info(
            "ONNX model precision=%s (graph io: input_type=%s output_type=%s)",
            model_precision,
            getattr(input_meta, "type", None),
            getattr(output_meta, "type", None),
        )
        logger.debug(
            "ONNX model io inputs=%s outputs=%s",
            [{"name": meta.name, "shape": meta.shape, "type": getattr(meta, "type", None)} for meta in inputs],
            [{"name": meta.name, "shape": meta.shape, "type": getattr(meta, "type", None)} for meta in outputs],
        )
        logger.debug(
            "ONNX graph io input_name=%s input_shape=%s input_type=%s output_shape=%s output_type=%s",
            input_meta.name,
            input_meta.shape,
            getattr(input_meta, "type", None),
            getattr(output_meta, "shape", None),
            getattr(output_meta, "type", None),
        )

    def run(self, inputs: Any) -> np.ndarray:
        """Run a forward pass on array or dictionary inputs."""
        if self._session is None:
            raise RuntimeError("ONNXBackend has not been loaded.")

        if isinstance(inputs, dict):
            input_feed = inputs
        else:
            array = np.asarray(inputs)
            input_feed = {self._input_name: array}

        try:
            with self._run_lock:
                outputs = self._session.run(self._output_names, input_feed)
        except Exception:
            logger.error("ONNX inference failed input_keys=%s", list(input_feed.keys()))
            raise

        # TODO: ModelPlugins should specify which output to use if multiple exist (embeddings, logits, etc)
        return outputs[0] if outputs else np.array([])

    def close(self) -> None:
        """Release runtime references so memory can be reclaimed promptly."""
        logger.debug("Closing ONNX backend")
        self._session = None
        self._input_name = ""
        self._providers = []
        self._requested_providers = []
        self._provider_options = []

    def output_names(self) -> list[str]:
        """Names of all output tensors (for debugging / advanced use)."""
        if self._session is None:
            return []
        return [o.name for o in self._session.get_outputs()]

    def input_shape(self) -> list[int]:
        """Expected input shape (from the model graph)."""
        if self._session is None:
            return []
        return list(self._session.get_inputs()[0].shape)

    @property
    def providers(self) -> list[str]:
        return self._providers

    @property
    def provider_options(self) -> list[dict[str, Any]]:
        return list(self._provider_options)

    def supports_true_batching(self) -> bool:
        """True batching is generally useful when a non-CPU provider is active."""
        return any(provider != "CPUExecutionProvider" for provider in self._providers)


# endregion ONNXBackend
