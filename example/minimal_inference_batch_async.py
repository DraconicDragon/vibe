import asyncio
import re
from pathlib import Path

import autotagger

MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"
IMAGE_FOLDER = Path("example/images/")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jxl"}


def _natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_images_from_folder(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise RuntimeError(f"Image folder does not exist: {folder}")

    # optional sort here because otherwise we get 1.jpg, 10.jpg, 11.jpg, ... before 2.jpg
    paths = [
        str(path)
        for path in sorted(folder.iterdir(), key=_natural_sort_key)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise RuntimeError(f"No supported images found in: {folder}")
    return paths


async def main() -> None:
    image_paths = _load_images_from_folder(IMAGE_FOLDER)

    # Using 'with' is optional but calls session.close() automatically to free resources when done.
    with autotagger.load("wd-eva02-large", source=MODEL_SOURCE, auto_download=False) as session:
        # Prepare inputs
        inputs = [(path, path) for path in image_paths]

        # Run async inference in batches
        batch_size = min(3, len(inputs))
        async for chunk in session.infer_async(inputs, batch_size=batch_size, batch_method="auto"):
            for item in chunk:
                # Result already sorted by score (high to low)
                score_dict = item.result.as_score_dict()

                top_3_scores = list(score_dict.items())[:3]

                # compact single-line output
                tags_str = ", ".join(f"{tag}:{score:.3f}" for tag, score in top_3_scores)
                print(f"{item.input_ref} -> {tags_str}")


if __name__ == "__main__":
    asyncio.run(main())