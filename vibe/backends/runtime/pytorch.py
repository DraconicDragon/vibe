"""
PyTorch inference backend.

Wraps a loaded torch model and provides a uniform .run(tensor) → ndarray
interface so the session layer doesn't need to know which backend is active.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe.precision import normalize_precision_string

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PyTorchBackend:
    """
    Loads and runs a PyTorch model.

    Supported weight formats:
      - .pt / .pth         (torch.load)
      - .safetensors       (safetensors.torch)

    The model is expected to be a full nn.Module saved with torch.save,
    or a state dict that can be loaded into a provided architecture.
    In practice, most HF tagger repos ship the full model — if you need
    state-dict loading, override _load_model in a plugin subclass.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._requested_precision: str = "auto"
        self._resolved_precision: str = "fp32"
        self._weight_dtype: Any = None
        self._compute_dtype: Any = None

    def load(self, weights_path: Path, device: str = "cpu", precision: str = "auto") -> None:
        """Load weights from disk. Raises if torch is not installed."""
        logger.debug("Loading PyTorch model")
        logger.debug("PyTorch weights path=%s", weights_path)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to use the pytorch backend. Install it with: pip install torch"
            ) from exc

        self._device = device
        self._requested_precision = normalize_precision_string(precision)
        suffix = weights_path.suffix.lower()

        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError(
                    "safetensors is required to load .safetensors weights. Install it with: pip install safetensors"
                ) from exc
            state = load_file(str(weights_path), device=device)
            # We just store the state dict here; the plugin is responsible
            # for building the architecture and calling load_state_dict.
            # See note in ModelPlugin.load_ancillary.
            self._model = state
            logger.debug("Loaded safetensors state dict device=%s", device)
        else:
            # .pt / .pth — attempt full model load first
            self._model = torch.load(
                weights_path,
                map_location=device,
                weights_only=False,
            )
            logger.debug("Loaded torch checkpoint device=%s", device)

        # Put in eval mode if it's an nn.Module
        try:
            import torch.nn as nn

            if isinstance(self._model, nn.Module):
                self._model.eval()
                self._model.to(device)
                self._apply_precision_plan(torch)
                try:
                    forward_params = list(inspect.signature(self._model.forward).parameters.keys())
                except Exception:
                    forward_params = []
                first_param = next(self._model.parameters(), None)
                if first_param is not None:
                    logger.debug(
                        "PyTorch model ready device=%s requested_precision=%s resolved_precision=%s weight_dtype=%s compute_dtype=%s",
                        device,
                        self._requested_precision,
                        self._resolved_precision,
                        first_param.dtype,
                        self._compute_dtype,
                    )
                logger.debug("PyTorch model input_names=%s", forward_params or ["input"])
                logger.debug("PyTorch model class=%s", self._model.__class__.__name__)
        except Exception:
            pass  # state dict case — handled by plugin

    @property
    def raw(self) -> Any:
        """Direct access to the loaded model/state dict for plugins that need it."""
        return self._model

    def run(self, tensor: Any) -> np.ndarray:
        """
        Run a forward pass.

        tensor should be a torch.Tensor of shape (1, C, H, W).
        Returns a numpy array of the model's output (after sigmoid if needed —
        that's the plugin's responsibility in postprocess).
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed.") from exc

        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        elif not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        if not isinstance(self._model, nn.Module):
            raise RuntimeError(
                "Model is a state dict, not an nn.Module. "
                "The plugin must build the architecture and call "
                "backend.raw to get the state dict, then construct "
                "the model itself."
            )

        with torch.no_grad():
            logger.debug("PyTorch run input_shape=%s input_dtype=%s", getattr(tensor, "shape", None), tensor.dtype)
            try:
                model_input = tensor.to(device=self._device, dtype=self._compute_dtype)
                output = self._model(model_input)
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
                    model_input = tensor.to(device=self._device, dtype=torch.float32)
                    output = self._model(model_input)
                else:
                    logger.error(
                        "PyTorch inference failed input_shape=%s device=%s",
                        getattr(tensor, "shape", None),
                        self._device,
                    )
                    raise

        if isinstance(output, torch.Tensor):
            logger.debug("PyTorch run output_shape=%s output_dtype=%s", output.shape, output.dtype)
            output_cpu = output.detach().cpu()
            if output_cpu.dtype == torch.bfloat16:
                # NumPy does not support bfloat16 tensors from PyTorch directly.
                output_cpu = output_cpu.to(dtype=torch.float32)
            return output_cpu.numpy()
        # Some models return tuples/lists
        if isinstance(output, (tuple, list)):
            logger.debug(
                "PyTorch run output tuple/list first_shape=%s first_dtype=%s",
                getattr(output[0], "shape", None),
                getattr(output[0], "dtype", None),
            )
            first = output[0]
            if isinstance(first, torch.Tensor):
                first_cpu = first.detach().cpu()
                if first_cpu.dtype == torch.bfloat16:
                    first_cpu = first_cpu.to(dtype=torch.float32)
                return first_cpu.numpy()
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
        """True batching is generally not useful for CPU device type."""
        return self._device != "cpu"

    def _apply_precision_plan(self, torch_module: Any) -> None:
        requested = self._requested_precision
        if requested == "int8_ov":
            requested = "auto"

        gpu_like_device = self._device.startswith(("cuda", "gpu", "mps"))
        has_cuda = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
        bf16_supported = False
        if has_cuda and callable(getattr(torch_module.cuda, "is_bf16_supported", None)):
            try:
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            except Exception:
                bf16_supported = False

        resolved = "fp32" if requested == "auto" else requested
        if resolved == "bf16" and gpu_like_device and self._device.startswith(("cuda", "gpu")) and not bf16_supported:
            has_fp16_gpu = has_cuda
            if has_fp16_gpu:
                logger.warning(
                    "Requested bf16 on device=%s but CUDA bf16 is unavailable; falling back to fp16.",
                    self._device,
                )
                resolved = "fp16"
            else:
                logger.warning(
                    "Requested bf16 on device=%s but accelerator bf16/fp16 support is unavailable; falling back to fp32.",
                    self._device,
                )
                resolved = "fp32"

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

        self._model.to(device=self._device, dtype=target_dtype)
        self._resolved_precision = resolved
        self._weight_dtype = target_dtype
        self._compute_dtype = target_dtype
        logger.info(
            "PyTorch precision configured requested=%s resolved=%s device=%s",
            self._requested_precision,
            self._resolved_precision,
            self._device,
        )
