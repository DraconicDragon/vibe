from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import vibe
from vibe.backends.base import Backend
from vibe.plugins.waifu_scorer import WaifuScorerBasePlugin, WaifuScorerV3Plugin


class _FakeClipModel(nn.Module):
    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        batch_size = int(images.shape[0])
        features = torch.ones((batch_size, 768), device=images.device, dtype=images.dtype)
        if normalize:
            return torch.nn.functional.normalize(features, dim=-1)
        return features


class _FakeBackend:
    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        self.raw = raw
        self.device = "cpu"
        self._model = None

    def _apply_precision_plan(self, torch_module: object) -> None:
        if self._model is not None:
            self._model.to(device=self.device, dtype=torch_module.float32)
            self._model.eval()


def test_registry_contains_waifu_scorer_models() -> None:
    models = set(vibe.list_models())
    assert "waifu-scorer-v3" in models
    assert "waifu-scorer-v4-beta" in models
    assert vibe.model_registry.get("waifu-scorer-v3") is WaifuScorerV3Plugin


def test_load_ancillary_builds_runtime_model_and_preprocesses(monkeypatch) -> None:
    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=lambda *args, **kwargs: (
            _FakeClipModel(),
            None,
            lambda image: torch.zeros(3, 4, 4),
        ),
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    plugin = WaifuScorerBasePlugin()
    backend = _FakeBackend(raw=plugin._build_mlp().state_dict())
    plugin.configure(backend=Backend.PYTORCH, backend_instance=backend)
    plugin.load_ancillary({})

    image = Image.new("RGB", (4, 4), (255, 0, 0))
    tensor = plugin.preprocess(image)
    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.dtype == torch.float32

    output = backend._model(tensor)
    assert output.shape == (1, 1)

    result = plugin.postprocess(output.detach().cpu().numpy())
    assert result.score_min == 0.0
    assert result.score_max == 10.0
    assert 0.0 <= result.score <= 10.0
