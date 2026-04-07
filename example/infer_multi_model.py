import asyncio
from pathlib import Path

import vibe

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODEL_A = {
    "name": "wd-eva02-large-v3",
    "source": "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3",
}
MODEL_B = {
    "name": "wd-swinv2-v3",
    "source": "SmilingWolf/wd-swinv2-tagger-v3",
    # "name": "wd-eva02-large-v3",
    # "source": "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3",
}


def load_images(folder: Path) -> list[str]:
    return [str(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


async def run_session(
    label: str,
    session: vibe.ModelSession,
    image_paths: list[str],
    *,
    batch_size: int = 4,
) -> None:
    # You can pass same images to both models, or different lists.
    inputs = [(path, path) for path in image_paths]

    async for chunk in session.infer_async(
        inputs,
        batch_size=batch_size,
        batch_method="auto",
    ):
        for item in chunk:
            score_dict = item.result.as_score_dict()
            top3 = list(score_dict.items())[:3]
            top3_str = ", ".join(f"{tag}:{score:.3f}" for tag, score in top3)
            print(f"[{label}] {item.input_ref} -> {top3_str}")


async def main() -> None:
    # Same images for both models:
    shared_images = load_images(Path("example/images"))

    # Or use different sets:
    # images_a = load_images(Path("example/images"))
    # images_b = load_images(Path("example/other_images"))

    with (
        vibe.load(MODEL_A["name"], source=MODEL_A["source"], backend="onnx", device="cuda") as session_a,
        vibe.load(MODEL_B["name"], source=MODEL_B["source"], backend="onnx", device="cuda") as session_b,
    ):
        await asyncio.gather(
            run_session("model-a", session_a, shared_images),
            run_session("model-b", session_b, shared_images),
        )


if __name__ == "__main__":
    asyncio.run(main())
