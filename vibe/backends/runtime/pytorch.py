"""
PyTorch inference backend.

Wraps a loaded torch model and provides a uniform .run(inputs) -> ndarray interface.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from vibe.backends.base import ExecutionPlan, ExecutionPreference, HardwareIntent
from vibe.config import config
from vibe.precision import PrecisionPolicy, PrecisionRequest, ResolvedPrecisionPlan

logger = logging.getLogger(__name__)


def _resolve_pytorch_device(preference: ExecutionPreference, torch_module: Any) -> str:
    if preference.intent == HardwareIntent.CPU:
        return "cpu"

    has_cuda = bool(getattr(torch_module.cuda, "is_available", lambda: False)())

    xpu_mod = getattr(torch_module, "xpu", None)
    has_xpu = bool(xpu_mod and callable(getattr(xpu_mod, "is_available", None)) and xpu_mod.is_available())

    has_mps = False
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend and callable(getattr(mps_backend, "is_available", None)):
        has_mps = bool(mps_backend.is_available())

    # User explicitly hinted a device class
    if preference.hint == "xpu":
        if not has_xpu:
            raise RuntimeError("Intel GPU (xpu) requested, but torch.xpu is not available.")
        return f"xpu:{preference.ordinal}" if preference.ordinal is not None else "xpu"

    if preference.intent == HardwareIntent.AUTO:
        if has_cuda:
            return "cuda"
        if has_xpu:
            return "xpu"
        if has_mps:
            return "mps"
        return "cpu"

    # Explicit ACCELERATOR requested (general)
    if has_cuda:
        return f"cuda:{preference.ordinal}" if preference.ordinal is not None else "cuda"
    if has_xpu:
        return f"xpu:{preference.ordinal}" if preference.ordinal is not None else "xpu"
    if has_mps:
        return "mps"

    raise RuntimeError(
        f"Accelerator requested ({preference.hint or 'gpu'}), but no CUDA, XPU, or MPS device is available in PyTorch."
    )


class PyTorchBackend:
    """
    Runs a fully constructed PyTorch `nn.Module`.

    Model construction and artifact interpretation are owned by the plugin via build_runtime().
    This class only owns framework placement and execution. Post-load, it is strictly immutable.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._plan: ResolvedPrecisionPlan | None = None
        self._weight_dtype: Any = None
        self._compute_dtype: Any = None
        self._autocast_device_type: str = "cpu"
        self._run_lock = threading.RLock()

    def load(self, model: Any, plan: ExecutionPlan) -> None:
        """Prepare a plugin-constructed module for execution."""
        started_at = time.perf_counter()
        logger.debug("Preparing PyTorch runtime")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to use the pytorch backend. Install it with: pip install torch"
            ) from exc

        # Configure cuDNN based on global config
        if not config.pytorch.cudnn_enabled:
            torch.backends.cudnn.enabled = False
            logger.info("cuDNN disabled via config")
        else:
            torch.backends.cudnn.enabled = True
            logger.debug("cuDNN enabled")

        self._model = model
        self._device = _resolve_pytorch_device(plan.preference, torch)

        if self._device.startswith("cuda"):
            self._autocast_device_type = "cuda"
        elif self._device.startswith("xpu"):
            self._autocast_device_type = "xpu"
        elif self._device.startswith("mps"):
            self._autocast_device_type = "mps"
        else:
            self._autocast_device_type = "cpu"

        if not isinstance(self._model, nn.Module):
            raise TypeError("PyTorchBackend requires a fully constructed torch.nn.Module.")

        self._model.eval()
        self._apply_precision_plan(torch, plan.precision)
        load_seconds = time.perf_counter() - started_at

        logger.debug(
            "PyTorch runtime ready in %.2fs class=%s device=%s plan=%s",
            load_seconds,
            self._model.__class__.__name__,
            self._device,
            self._plan,
        )

    def execution_info(self) -> dict[str, Any]:
        """Return runtime-reported diagnostics."""
        return {
            "device": self._device,
            "autocast_device_type": self._autocast_device_type,
            "precision": self._plan.to_dict() if self._plan else None,
        }

    def run(self, inputs: Any) -> Any:
        """Run a forward pass on generic tensor, tuple, or dict inputs."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed.") from exc

        if not isinstance(self._model, nn.Module) or self._plan is None:
            raise TypeError("PyTorchBackend has no loaded torch.nn.Module.")

        args, kwargs = self._prepare_inputs(inputs, torch, self._plan)

        # Enforce strict thread safety for pooled backends.
        with self._run_lock, torch.no_grad():
            if self._plan.autocast_enabled:
                with torch.autocast(device_type=self._autocast_device_type, dtype=self._compute_dtype):
                    output = self._model(*args, **kwargs)
            else:
                output = self._model(*args, **kwargs)

        return self._tensor_to_numpy(output, torch)

    def _prepare_inputs(
        self, inputs: Any, torch_module: Any, plan: ResolvedPrecisionPlan
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Generic input unpacker for Tensor, tuple, dict, or structured object."""

        def _to_dev(val: Any) -> Any:
            if isinstance(val, np.ndarray):
                val = torch_module.from_numpy(val)
            if isinstance(val, torch_module.Tensor):
                if val.dtype.is_floating_point:
                    # If autocast is off, inputs must match weight dtype
                    if not plan.autocast_enabled and self._weight_dtype is not None:
                        return val.to(device=self._device, dtype=self._weight_dtype)
                    # If autocast is on, leave as float32 on the device
                    return val.to(device=self._device)
                return val.to(device=self._device)
            return val

        if isinstance(inputs, dict):
            return (), {k: _to_dev(v) for k, v in inputs.items()}

        if isinstance(inputs, (tuple, list)):
            return tuple(_to_dev(v) for v in inputs), {}

        # Structured container with named fields (e.g. NaFlex batch patches/sizes)
        if hasattr(inputs, "_fields"):
            args = tuple(_to_dev(getattr(inputs, field_name)) for field_name in inputs._fields)
            return args, {}

        return (_to_dev(inputs),), {}

    def _tensor_to_numpy(self, output: Any, torch_module: Any) -> Any:
        if isinstance(output, torch_module.Tensor):
            out = output.detach().cpu()
            if out.dtype in (torch_module.bfloat16, torch_module.float16):
                out = out.to(torch_module.float32)
            return out.numpy()

        # Preserve dictionary outputs
        if isinstance(output, dict):
            return {k: self._tensor_to_numpy(v, torch_module) for k, v in output.items()}

        # Preserve tuple/list outputs natively instead of taking [0]
        if isinstance(output, (tuple, list)):
            return type(output)(self._tensor_to_numpy(v, torch_module) for v in output)

        return np.array(output)

    def close(self) -> None:
        """Release model references and clear GPU cache."""
        logger.debug("Closing PyTorch backend device=%s", self._device)
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def device(self) -> str:
        return self._device

    def supports_true_batching(self) -> bool:
        """Return whether the current device type should use true batching."""
        return self._device != "cpu"

    def _apply_precision_plan(self, torch_module: Any, request: PrecisionRequest) -> None:
        has_cuda = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
        xpu_mod = getattr(torch_module, "xpu", None)
        has_xpu = bool(xpu_mod and callable(getattr(xpu_mod, "is_available", None)) and xpu_mod.is_available())

        bf16_supported = False
        if has_cuda and callable(getattr(torch_module.cuda, "is_bf16_supported", None)):
            try:
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            except Exception:
                pass
        elif has_xpu and callable(getattr(xpu_mod, "is_bf16_supported", None)):
            try:
                bf16_supported = bool(xpu_mod.is_bf16_supported())
            except Exception:
                pass

        # 1. Resolve Compute Policy
        compute_policy = request.compute
        if compute_policy == PrecisionPolicy.AUTO:
            if self._autocast_device_type == "cuda":
                compute_policy = PrecisionPolicy.BF16 if bf16_supported else PrecisionPolicy.FP16
            else:
                compute_policy = PrecisionPolicy.FP32

        # 2. Hardware Fallbacks for Compute
        if compute_policy == PrecisionPolicy.BF16 and not bf16_supported:
            if request.fallback_allowed:
                # CPU should fall back to fp32. GPUs fall back to fp16.
                fallback_target = (
                    PrecisionPolicy.FP16 if self._autocast_device_type in ("cuda", "mps") else PrecisionPolicy.FP32
                )
                logger.warning(
                    "Device '%s' does not support bfloat16 natively. Falling back compute to %s.",
                    self._device,
                    fallback_target.value,
                )
                compute_policy = fallback_target
            else:
                raise RuntimeError(f"Strict precision request failed: Device {self._device} does not support bf16.")

        # 3. Resolve Weight Policy
        weight_policy = request.weight
        if weight_policy == PrecisionPolicy.AUTO:
            weight_policy = compute_policy

        dtype_map = {
            PrecisionPolicy.FP32: torch_module.float32,
            PrecisionPolicy.FP16: torch_module.float16,
            PrecisionPolicy.BF16: torch_module.bfloat16,
        }

        self._compute_dtype = dtype_map.get(compute_policy, torch_module.float32)
        self._weight_dtype = dtype_map.get(weight_policy)  # None if PRESERVE

        # 4. Apply Weights
        autocast_needed = self._compute_dtype != torch_module.float32

        if hasattr(self._model, "apply_precision"):
            logger.debug("Delegating precision casting to model custom apply_precision hook")
            custom_autocast = self._model.apply_precision(
                device=self._device,
                dtype=self._weight_dtype or torch_module.float32,
                requested=request,
                bf16_supported=bf16_supported,
            )
            # If the model explicitly returns a boolean, respect its absolute authority over execution
            if isinstance(custom_autocast, bool):
                autocast_needed = custom_autocast
        elif self._weight_dtype is not None:
            self._model.to(device=self._device, dtype=self._weight_dtype)
        else:
            self._model.to(device=self._device)

        # 5. Lock in the plan
        self._plan = ResolvedPrecisionPlan(
            weight_dtype=str(self._weight_dtype) if self._weight_dtype else "preserve",
            compute_dtype=str(self._compute_dtype),
            autocast_enabled=autocast_needed,
        )
        logger.debug(
            "PyTorch precision initialized: request=(%s, %s) -> resolved=(weight=%s, compute=%s, autocast=%s)",
            request.weight.value,
            request.compute.value,
            self._plan.weight_dtype,
            self._plan.compute_dtype,
            self._plan.autocast_enabled,
        )
