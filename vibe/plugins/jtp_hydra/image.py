from typing import Any, Callable, TypeAlias

import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from PIL.ImageOps import exif_transpose
from torch import Tensor

# Types and Classes for official model.py imports
Source: TypeAlias = PILImage.Image | bytes | str
Size: TypeAlias = tuple[int, int]
Box: TypeAlias = tuple[int, int, int, int]
Color: TypeAlias = tuple[int | float, ...] | int | float
ResizeFn: TypeAlias = Callable[[Size], Size | None]
CropFn: TypeAlias = Callable[[Size], Box | None]
Image: TypeAlias = PILImage.Image


class Kernel:
    LANCZOS3 = 1
    MKS2013 = 2


def open_srgb(
    source: Source,
    *,
    expect: Size | None = None,
    crop: CropFn | Box | None = None,
    resize: ResizeFn | Size | None = None,
    background: Color = 0,
    kernel: Any = None,
    linear: bool = False,
) -> PILImage.Image:
    if isinstance(source, PILImage.Image):
        img = source
    elif isinstance(source, bytes):
        import io

        img = PILImage.open(io.BytesIO(source))
    else:
        img = PILImage.open(str(source))

    img.load()
    try:
        exif_transpose(img, in_place=True)
    except Exception:
        pass

    size = (img.height, img.width)

    if expect is not None and size != expect:
        raise RuntimeError(f"Image is {size[1]}x{size[0]}, but expected {expect[1]}x{expect[0]}.")

    if crop is not None and not isinstance(crop, tuple):
        crop = crop(size)

    if crop is not None:
        left, top, right, bottom = crop
        img = img.crop((left, top, right, bottom))
        size = (img.height, img.width)

    if resize is not None and not isinstance(resize, tuple):
        resize = resize(size)

    # Transparency/background alpha flattening
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg_color = background if isinstance(background, tuple) else (background, background, background)
        bg = PILImage.new("RGB", img.size, bg_color)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if resize is not None and size != resize:
        img = img.resize((resize[1], resize[0]), PILImage.Resampling.LANCZOS)

    return img


def as_tensor(img: Any) -> Tensor:
    return torch.from_numpy(np.asarray(img))


def put(img: Any, tensor: Tensor) -> None:
    np.copyto(tensor.numpy(), img, casting="no")


def put_patches(img: Any, patches: Tensor, patch_size: int, **kwargs: Any) -> Size:
    return (0, 0)


def spread(img: Tensor, patch_size: int) -> Tensor:
    return rearrange(img, "... (h p1) (w p2) c -> ... h w p1 p2 c", p1=patch_size, p2=patch_size)


def patchify(
    img: Any | Tensor, patch_size: int, *, ensure_batch_dim: bool = True, dtype: torch.dtype | None = None
) -> Tensor:
    if not isinstance(img, Tensor):
        img = as_tensor(img)

    if ensure_batch_dim and img.ndim == 3:
        img = img.unsqueeze(0)

    img = spread(img, patch_size)

    if dtype is not None and dtype != img.dtype:
        img = img.to(dtype=dtype, memory_format=torch.contiguous_format)

    return img.flatten(-3)


def stack(
    images: list[Tensor],
    patch_size: int,
    max_seq: int,
    *,
    max_n: int = 0,
    channels: int = 3,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if device is None:
        device = images[0].device
    if dtype is None:
        dtype = images[0].dtype

    batch = torch.empty(len(images), max_seq, patch_size * patch_size * channels, device=device, dtype=dtype)
    sizes = torch.empty(len(images), 2, device="cpu", dtype=torch.uint16)

    if max_n > 1:
        torch._dynamo.mark_dynamic(batch, 0, min=1, max=max_n)
        torch._dynamo.mark_dynamic(sizes, 0, min=1, max=max_n)

    srcs: list[Tensor] = []
    dests: list[Tensor] = []
    zero: list[Tensor] = []
    for idx, img in enumerate(images):
        img = spread(img, patch_size)
        assert img.ndim == 5

        h, w = img.shape[:2]
        sizes[idx, 0] = h
        sizes[idx, 1] = w
        seqlen = h * w

        srcs.append(img)
        dests.append(batch[idx, :seqlen].view(img.shape))

        if seqlen < max_seq:
            zero.append(batch[idx, seqlen:])

    torch._foreach_copy_(dests, srcs, non_blocking=True)
    if zero:
        torch._foreach_zero_(zero)

    return batch, sizes
