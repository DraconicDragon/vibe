import logging
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# DINOv3 ViT-H/16+

D_MODEL = 1280
N_HEADS = 20
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 32
D_FFN = 5120
N_REGISTERS = 4
PATCH_SIZE = 16
ROPE_THETA = 100.0
ROPE_RESCALE = 2.0
LN_EPS = 1e-5
LAYERSCALE = 1.0

FEATURE_DIM = (1 + N_REGISTERS) * D_MODEL  # 6400


@lru_cache(maxsize=32)
def _patch_coords_cached(h: int, w: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    coords = coords * ROPE_RESCALE
    return coords


def _build_rope(h_patches: int, w_patches: int, dtype: torch.dtype, device: torch.device):
    coords = _patch_coords_cached(h_patches, w_patches, str(device))
    inv_freq = 1.0 / (ROPE_THETA ** torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32, device=device))
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    n_pre = 1 + N_REGISTERS
    q_pre, q_pat = q[..., :n_pre, :], q[..., n_pre:, :]
    k_pre, k_pat = k[..., :n_pre, :], k[..., n_pre:, :]
    q_pat = q_pat * cos + _rotate_half(q_pat) * sin
    k_pat = k_pat * cos + _rotate_half(k_pat) * sin
    return torch.cat([q_pre, q_pat], dim=-2), torch.cat([k_pre, k_pat], dim=-2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, D_MODEL))


class _GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.up_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.down_proj = nn.Linear(D_FFN, D_MODEL, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attention = _Attention()
        self.layer_scale1 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.mlp = _GatedMLP()
        self.layer_scale2 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin) * self.layer_scale1
        x = x + self.mlp(self.norm2(x)) * self.layer_scale2
        return x


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.register_tokens = nn.Parameter(torch.zeros(1, N_REGISTERS, D_MODEL))
        self.patch_embeddings = nn.Conv2d(3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(pixel_values.to(dtype)).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        return torch.cat([cls, regs, patches], dim=1)


class DINOv3ViTH(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layer = nn.ModuleList([_Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL, eps=LN_EPS)

    def forward(self, pixel_values):
        _, _, H, W = pixel_values.shape
        x = self.embeddings(pixel_values)
        h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
        cos, sin = _build_rope(h_p, w_p, x.dtype, pixel_values.device)
        for block in self.layer:
            x = block(x, cos, sin)
        return self.norm(x)


# Head auto-detection


class _LowRankHead(nn.Module):
    def __init__(self, in_dim: int, rank: int, num_tags: int, down_bias: bool, up_bias: bool):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, rank, bias=down_bias)
        self.proj_up = nn.Linear(rank, num_tags, bias=up_bias)

    def forward(self, x):
        return self.proj_up(self.proj_down(x))


def _build_head_from_checkpoint(head_sd: dict, in_dim: int, num_tags: int) -> tuple[nn.Module, dict]:
    weights_2d = [(k, v) for k, v in head_sd.items() if k.endswith(".weight") and v.ndim == 2]

    # Case 1: single dense linear
    singles = [(k, v) for k, v in weights_2d if tuple(v.shape) == (num_tags, in_dim)]
    if len(weights_2d) <= 2 and len(singles) == 1:
        wkey, wval = singles[0]
        base = wkey[: -len(".weight")]
        bias_key = base + ".bias"
        has_bias = bias_key in head_sd
        module = nn.Linear(in_dim, num_tags, bias=has_bias)
        remapped = {"weight": wval}
        if has_bias:
            remapped["bias"] = head_sd[bias_key]
        return module, remapped

    # Case 2: low-rank pair
    down, up = None, None
    for k, v in weights_2d:
        if v.shape[1] == in_dim and v.shape[0] != num_tags:
            down = (k, v)
        elif v.shape[0] == num_tags and v.shape[1] != in_dim:
            up = (k, v)

    if down is not None and up is not None:
        rank_down, rank_up = down[1].shape[0], up[1].shape[1]
        if rank_down != rank_up:
            raise RuntimeError(f"Low-rank head: inner dims disagree (down out={rank_down}, up in={rank_up})")

        has_down_bias = (down[0][: -len(".weight")] + ".bias") in head_sd
        has_up_bias = (up[0][: -len(".weight")] + ".bias") in head_sd

        module = _LowRankHead(in_dim, rank_down, num_tags, has_down_bias, has_up_bias)
        remapped = {"proj_down.weight": down[1], "proj_up.weight": up[1]}
        if has_down_bias:
            remapped["proj_down.bias"] = head_sd[down[0][: -len(".weight")] + ".bias"]
        if has_up_bias:
            remapped["proj_up.bias"] = head_sd[up[0][: -len(".weight")] + ".bias"]

        logger.info(f"Detected low-rank head: rank={rank_down}, num_tags={num_tags}")
        return module, remapped

    raise RuntimeError("Could not infer head architecture from checkpoint.")


class DINOv3Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv3ViTH()
        self.head: nn.Module | None = None

    def apply_precision(
        self,
        device: str,
        dtype: torch.dtype,
        requested: str = "auto",
        bf16_supported: bool = False,
    ) -> None:
        if requested == "auto":
            if device == "cpu":
                backbone_dtype = torch.float32
            elif bf16_supported:
                backbone_dtype = torch.bfloat16
            else:
                backbone_dtype = torch.float16
            head_dtype = torch.float32
        else:
            backbone_dtype = dtype
            head_dtype = dtype

        self.backbone.to(device=device, dtype=backbone_dtype)
        if self.head is not None:
            self.head.to(device=device, dtype=head_dtype)

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values)
        cls = hidden[:, 0, :]
        regs = hidden[:, 1 : 1 + N_REGISTERS, :].flatten(1)

        # Features come out in whatever dtype the backbone is using (bf16)
        features = torch.cat([cls, regs], dim=-1)

        # Convert features to match the head's precision (fp32)
        if self.head is not None:
            try:
                head_dtype = next(self.head.parameters()).dtype
                features = features.to(head_dtype)
            except StopIteration:
                pass

        return self.head(features)


def split_and_clean_state_dict(sd: dict) -> tuple[dict, dict]:
    backbone_sd, head_sd = {}, {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            nk = k[len("backbone.") :]
            if nk.startswith("model.layer."):
                nk = nk[len("model.") :]
            backbone_sd[nk] = v
        else:
            head_sd[k] = v

    for k in list(backbone_sd.keys()):
        if ".layer_scale" in k and k.endswith(".lambda1"):
            backbone_sd[k[: -len(".lambda1")]] = backbone_sd.pop(k)
        if "rope_embeddings" in k:
            backbone_sd.pop(k)

    return backbone_sd, head_sd
