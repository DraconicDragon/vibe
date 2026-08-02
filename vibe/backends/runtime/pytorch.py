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

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """
    Runs a fully constructed PyTorch `nn.Module`.

    Model construction and artifact interpretation are owned by the plugin via build_runtime().
    This class only owns framework placement and execution.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._requested_precision: str = "auto"
        self._resolved_precision: str = "fp32"
        self._weight_dtype: Any = None
        self._compute_dtype: Any = None
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
        self._requested_precision = request.precision
        if not isinstance(self._model, nn.Module):
            raise TypeError("PyTorchBackend requires a fully constructed torch.nn.Module.")

        self._model.eval()
        self._model.to(self._device)
        self._apply_precision_plan(torch)

        logger.debug(
            "Attached pre-built model class=%s device=%s weight_dtype=%s compute_dtype=%s",
            self._model.__class__.__name__,
            self._device,
            self._weight_dtype,
            self._compute_dtype,
        )

    def run(self, inputs: Any) -> np.ndarray:
        """Run a forward pass on generic tensor, tuple, or dict inputs."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed.") from exc

        if not isinstance(self._model, nn.Module):
            raise TypeError("PyTorchBackend has no loaded torch.nn.Module.")

        args, kwargs = self._prepare_inputs(inputs, torch)

        with self._run_lock, torch.no_grad():
            try:
                output = self._model(*args, **kwargs)
            except Exception:
                if self._compute_dtype != torch.float32:
                    logger.warning(
                        "PyTorch inference failed with compute_dtype=%s on device=%s; retrying with float32 fallback.",
                        self._compute_dtype,
                        self._device,
                    )
                    self._compute_dtype = torch.float32
                    self._resolved_precision = "fp32"
                    self._model.to(device=self._device, dtype=torch.float32)
                    args, kwargs = self._prepare_inputs(inputs, torch)
                    output = self._model(*args, **kwargs)
                else:
                    logger.error(
                        "PyTorch inference failed args[0].shape=%s device=%s",
                        getattr(args[0], "shape", None),
                        self._device,
                    )
                    raise

        return self._tensor_to_numpy(output, torch)

    def _prepare_inputs(self, inputs: Any, torch_module: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Generic input unpacker for Tensor, tuple, dict, or structured object."""

        def _to_dev(val: Any) -> Any:
            if isinstance(val, np.ndarray):
                val = torch_module.from_numpy(val)
            if isinstance(val, torch_module.Tensor):
                if val.dtype.is_floating_point:
                    return val.to(device=self._device, dtype=self._compute_dtype)
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
            if out.dtype == torch_module.bfloat16:
                out = out.to(torch_module.float32)
            return out.numpy()

        if isinstance(output, (tuple, list)):
            first = output[0]
            if isinstance(first, torch_module.Tensor):
                first = first.detach().cpu()
                if first.dtype == torch_module.bfloat16:
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

    def _apply_precision_plan(self, torch_module: Any) -> None:
        requested = self._requested_precision
        if requested == "int8_ov":
            requested = "auto"

        has_cuda = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
        bf16_supported = False
        if has_cuda and callable(getattr(torch_module.cuda, "is_bf16_supported", None)):
            try:
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            except Exception:
                bf16_supported = False

        resolved = "fp32" if requested == "auto" else requested

        # If bf16 is requested on a CUDA/GPU device but not supported, fall back to fp16
        if resolved == "bf16" and self._device.startswith(("cuda", "gpu")) and not bf16_supported:
            logger.warning(
                "Requested bf16 on device=%s but CUDA bf16 is unavailable; falling back to fp16.",
                self._device,
            )
            resolved = "fp16"

        dtype_map = {
            "fp32": torch_module.float32,
            "fp16": torch_module.float16,
            "bf16": torch_module.bfloat16,
        }
        target_dtype = dtype_map.get(resolved, torch_module.float32)

        if hasattr(self._model, "apply_precision"):
            logger.debug("Delegating precision casting to model custom apply_precision hook")
            self._model.apply_precision(
                device=self._device,
                dtype=target_dtype,
                requested=self._requested_precision,
                bf16_supported=bf16_supported,
            )
        else:
            self._model.to(device=self._device, dtype=target_dtype)

        self._resolved_precision = resolved
        self._weight_dtype = target_dtype
        self._compute_dtype = torch_module.float32
        logger.info(
            "PyTorch precision configured requested=%s resolved=%s device=%s weight_dtype=%s compute_dtype=%s",
            self._requested_precision,
            self._resolved_precision,
            self._device,
            self._weight_dtype,
            self._compute_dtype,
        )
