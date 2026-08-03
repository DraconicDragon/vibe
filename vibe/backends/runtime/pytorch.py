"""
PyTorch inference backend.

Wraps a loaded torch model and provides a uniform .run(inputs) -> ndarray interface.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from vibe.backends.base import ExecutionRequest
from vibe.precision import PrecisionPolicy, PrecisionRequest, ResolvedPrecisionPlan

logger = logging.getLogger(__name__)


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

    def load(self, model: Any, request: ExecutionRequest) -> None:
        """Prepare a plugin-constructed module for execution."""
        logger.debug("Preparing PyTorch runtime")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to use the pytorch backend. Install it with: pip install torch"
            ) from exc

        # Configure cuDNN based on environment variable
        if os.getenv("VIBE_DISABLE_CUDNN", "false").lower() in ("true", "1", "yes"):
            torch.backends.cudnn.enabled = False
            logger.info("cuDNN disabled via VIBE_DISABLE_CUDNN environment variable")
        else:
            torch.backends.cudnn.enabled = True
            logger.debug("cuDNN enabled (VIBE_DISABLE_CUDNN not set or false)")

        self._model = model
        self._device = request.device

        if self._device.startswith(("cuda", "gpu")):
            self._autocast_device_type = "cuda"
        elif self._device.startswith("mps"):
            self._autocast_device_type = "mps"
        else:
            self._autocast_device_type = "cpu"

        if not isinstance(self._model, nn.Module):
            raise TypeError("PyTorchBackend requires a fully constructed torch.nn.Module.")

        self._model.eval()
        self._apply_precision_plan(torch, request.precision)

        logger.debug(
            "Attached pre-built model class=%s device=%s plan=%s",
            self._model.__class__.__name__,
            self._device,
            self._plan,
        )

    def run(self, inputs: Any) -> np.ndarray:
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
                    # If autocast is off, inputs must exactly match the model's weight dtype
                    if not plan.autocast_enabled and self._weight_dtype is not None:
                        return val.to(device=self._device, dtype=self._weight_dtype)
                    # If autocast is on, leave it as fp32 on the correct device; autocast handles it internally
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

    def _tensor_to_numpy(self, output: Any, torch_module: Any) -> np.ndarray:
        if isinstance(output, torch_module.Tensor):
            out = output.detach().cpu()
            if out.dtype in (torch_module.bfloat16, torch_module.float16):
                out = out.to(torch_module.float32)
            return out.numpy()

        if isinstance(output, (tuple, list)):
            first = output[0]
            if isinstance(first, torch_module.Tensor):
                first = first.detach().cpu()
                if first.dtype in (torch_module.bfloat16, torch_module.float16):
                    first = first.to(torch_module.float32)
                return first.numpy()
            return np.array(first)

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
        bf16_supported = False
        if has_cuda and callable(getattr(torch_module.cuda, "is_bf16_supported", None)):
            try:
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
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
                logger.warning(
                    "Device '%s' does not support bfloat16 natively. Falling back compute to fp16.",
                    self._device,
                )
                compute_policy = PrecisionPolicy.FP16
            else:
                raise RuntimeError(f"Strict precision request failed: Device {self._device} does not support bf16.")

        # 3. Resolve Weight Policy
        weight_policy = request.weight
        if weight_policy == PrecisionPolicy.AUTO:
            # Historically 'auto' meant 'cast weights to match compute' to save VRAM
            weight_policy = compute_policy

        dtype_map = {
            PrecisionPolicy.FP32: torch_module.float32,
            PrecisionPolicy.FP16: torch_module.float16,
            PrecisionPolicy.BF16: torch_module.bfloat16,
        }

        self._compute_dtype = dtype_map.get(compute_policy, torch_module.float32)
        self._weight_dtype = dtype_map.get(weight_policy)  # None if PRESERVE

        # 4. Apply Weights
        if self._weight_dtype is not None:
            if hasattr(self._model, "apply_precision"):
                logger.debug("Delegating precision casting to model custom apply_precision hook")
                self._model.apply_precision(
                    device=self._device,
                    dtype=self._weight_dtype,
                    requested=request,
                    bf16_supported=bf16_supported,
                )
            else:
                self._model.to(device=self._device, dtype=self._weight_dtype)
        else:
            self._model.to(device=self._device)

        # 5. Lock in the plan
        self._plan = ResolvedPrecisionPlan(
            weight_dtype=str(self._weight_dtype) if self._weight_dtype else "preserve",
            compute_dtype=str(self._compute_dtype),
            autocast_enabled=(self._compute_dtype != torch_module.float32),
        )
        logger.info(
            "PyTorch precision initialized: request=(%s, %s) -> resolved=(weight=%s, compute=%s, autocast=%s)",
            request.weight.value,
            request.compute.value,
            self._plan.weight_dtype,
            self._plan.compute_dtype,
            self._plan.autocast_enabled,
        )
