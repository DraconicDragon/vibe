"""
PyTorch inference backend.

Wraps a loaded torch model and provides a uniform .run(tensor) → ndarray
interface so the session layer doesn't need to know which backend is active.
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

    Model construction and artifact interpretation are deliberately owned by
    the plugin. This class only owns framework placement and execution.
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

    def run(self, tensor: Any) -> np.ndarray:
        """Run a forward pass. Returns a numpy array of raw model output.

        Accepts either:
        - a standard torch.Tensor / ndarray for normal single-input models
        - a JTPHydraBatch (NamedTuple with patches/sizes) for
            the NaFlex multi-input forward pass
        """
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed.") from exc

        # Lazy import to avoid circular dependency at module level.
        from vibe.plugins.jtp_hydra.jtp_hydra_modelplugin import JTPHydraBatch

        if not isinstance(self._model, nn.Module):
            raise TypeError("PyTorchBackend has no loaded torch.nn.Module.")

        if isinstance(tensor, JTPHydraBatch):
            # NaFlex three-input forward pass.
            patches = tensor.patches if tensor.patches.ndim == 3 else tensor.patches.unsqueeze(0)
            sizes = tensor.sizes if tensor.sizes.ndim == 2 else tensor.sizes.unsqueeze(0)
            logger.debug(
                "PyTorch JTP-3 / Hydra run batch_size=%s patches_shape=%s sizes_shape=%s",
                patches.shape[0] if patches.ndim > 0 else None,
                patches.shape,
                sizes.shape,
            )
            p = patches.to(device=self._device, dtype=self._compute_dtype).div(127.5).sub(1.0)
            sz = sizes.to(device=self._device, dtype=torch.int32)
            args = (p, sz)
        else:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            elif not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            logger.debug("PyTorch run input_shape=%s input_dtype=%s", tensor.shape, tensor.dtype)
            args = (tensor.to(device=self._device, dtype=self._compute_dtype),)

        with self._run_lock, torch.no_grad():
            try:
                output = self._model(*args)
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
                    args = tuple(
                        a.to(dtype=torch.float32) if isinstance(a, torch.Tensor) and a.dtype.is_floating_point else a
                        for a in args
                    )
                    output = self._model(*args)
                elif self._weight_dtype not in {None, torch.float32}:
                    logger.debug(
                        "FP32 compute with weight_dtype=%s on device=%s; temporarily promoting weights.",
                        self._weight_dtype,
                        self._device,
                    )
                    original_weight_dtype = self._weight_dtype
                    self._model.to(device=self._device, dtype=torch.float32)
                    try:
                        output = self._model(*args)
                    finally:
                        self._model.to(device=self._device, dtype=original_weight_dtype)
                else:
                    logger.error(
                        "PyTorch inference failed args[0].shape=%s device=%s",
                        getattr(args[0], "shape", None),
                        self._device,
                    )
                    raise

        if isinstance(output, torch.Tensor):
            logger.debug("PyTorch run output_shape=%s output_dtype=%s", output.shape, output.dtype)
            out = output.detach().cpu()
            if out.dtype == torch.bfloat16:
                out = out.to(torch.float32)
            return out.numpy()
        if isinstance(output, (tuple, list)):
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = first.detach().cpu()
                if first.dtype == torch.bfloat16:
                    first = first.to(torch.float32)
                return first.numpy()
            return np.array(first)
        return np.array(output)

    def close(self) -> None:
        """Release model references and try to free backend-side cache."""
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

        if self._device == "cpu" and resolved in {"fp16", "bf16"}:
            logger.info(
                "Requested %s on CPU; keeping requested weight/compute dtype where possible. "
                "If ops are unsupported at runtime, backend will retry with fp32.",
                resolved,
            )

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
