"""
PyTorch inference backend.

Wraps a loaded torch model and provides a uniform .run(tensor) → ndarray
interface so the session layer doesn't need to know which backend is active.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


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

    def load(self, weights_path: Path, device: str = "cpu") -> None:
        """Load weights from disk. Raises if torch is not installed."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to use the pytorch backend. Install it with: pip install torch"
            ) from exc

        self._device = device
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
        else:
            # .pt / .pth — attempt full model load first
            self._model = torch.load(
                weights_path,
                map_location=device,
                weights_only=False,
            )

        # Put in eval mode if it's an nn.Module
        try:
            import torch.nn as nn

            if isinstance(self._model, nn.Module):
                self._model.eval()
                self._model.to(device)
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

        if not isinstance(self._model, nn.Module):
            raise RuntimeError(
                "Model is a state dict, not an nn.Module. "
                "The plugin must build the architecture and call "
                "backend.raw to get the state dict, then construct "
                "the model itself."
            )

        with torch.no_grad():
            output = self._model(tensor.to(self._device))

        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        # Some models return tuples/lists
        if isinstance(output, (tuple, list)):
            return output[0].cpu().numpy()
        return np.array(output)

    def close(self) -> None:
        """Release model references and try to free backend-side cache."""
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
