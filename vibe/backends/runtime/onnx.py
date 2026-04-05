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
from pathlib import Path
from typing import Any

import numpy as np

from vibe.devices import normalize_device_string

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
    value = normalize_device_string(device, backend="onnx")
    if not value or value == "auto":
        return True, 0
    if value == "cpu":
        return False, 0
    if value in {"gpu", "rocm", "dml"}:
        return True, 0

    for prefix in ("gpu:", "rocm:", "dml:"):
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
        self._providers: list[str] = []
        self._provider_options: list[dict[str, Any]] = []

    def load(
        self,
        weights_path: Path,
        providers: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Load an ONNX model. Raises if onnxruntime is not installed.

        providers:   Override the provider list.
                 If omitted, resolves from env and device preference.
        device:      Logical device selector for auto-provider selection
                 (e.g. "cpu", "gpu", "gpu1", "cuda:0"). 'cuda' and 'gpu' are interchangeable.
        """
        logger.debug("Loading ONNX model from %s", weights_path)
        prepare_onnxruntime_environment()

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to use the onnx backend. "
                "Install it with: pip install onnxruntime  "
                "(or onnxruntime-gpu for CUDA support)"
            ) from exc

        resolved_providers, resolved_provider_options = resolve_onnx_provider_chain(
            device=device,
            requested_providers=providers,
            ort_module=ort,
        )

        self._providers = resolved_providers
        self._provider_options = resolved_provider_options or []
        self._session = ort.InferenceSession(
            str(weights_path),
            providers=resolved_providers,
            provider_options=resolved_provider_options,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_name = inputs[0].name
        input_meta = inputs[0]
        output_meta = outputs[0] if outputs else None
        primary_provider = self._providers[0] if self._providers else "CPUExecutionProvider"
        fallback_providers = self._providers[1:]
        if fallback_providers:
            logger.info(
                "ONNX model loaded: using %s with fallback %s",
                primary_provider,
                fallback_providers,
            )
        else:
            logger.info("ONNX model loaded: using %s", primary_provider)
        logger.debug(
            "ONNX precision input_type=%s output_type=%s weight_precision=graph-defined compute_precision=runtime-defined casted_weight_precision=no casted_compute_precision=no",
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

    def run(self, array: np.ndarray) -> np.ndarray:
        """
        Run a forward pass.

        array should be a float32 numpy array of shape (1, C, H, W)
        or (1, H, W, C) depending on the model — the plugin's preprocess
        method is responsible for the correct layout.

        Returns the first output tensor as a numpy array.
        """
        if self._session is None:
            raise RuntimeError("ONNXBackend.load() has not been called.")

        logger.debug("ONNX run input_shape=%s input_dtype=%s", array.shape, array.dtype)
        if array.ndim == 0:
            logger.error(
                "ONNX input is scalar for input_name=%s expected_shape=%s actual_shape=%s",
                self._input_name,
                self.input_shape(),
                array.shape,
            )

        # TODO: support selecting a specific output name/index per model to avoid
        # fetching all outputs when only one tensor is needed.
        try:
            outputs = self._session.run(None, {self._input_name: array})
        except Exception:
            logger.error(
                "ONNX inference failed input_name=%s input_shape=%s expected_input_shape=%s",
                self._input_name,
                array.shape,
                self.input_shape(),
            )
            raise
        if outputs:
            first = outputs[0]
            logger.debug(
                "ONNX run output_shape=%s output_dtype=%s",
                getattr(first, "shape", None),
                getattr(first, "dtype", None),
            )
        return outputs[0]

    def close(self) -> None:
        """Release runtime references so memory can be reclaimed promptly."""
        logger.debug("Closing ONNX backend")
        self._session = None
        self._input_name = ""
        self._providers = []
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
