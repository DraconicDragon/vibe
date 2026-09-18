import logging

import vibe

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("vibe").setLevel(logging.WARNING)

MLP_SOURCE = "/home/drac/dev/models/seperate/waifu-scorer-v3/"
CLIP_SOURCE = "/home/drac/dev/models/seperate/clip-vit-large-patch14/"

with vibe.load(
    "waifu-scorer-v3",
    source=MLP_SOURCE,
    source_map={
        "clip_weights": CLIP_SOURCE,
        "clip_config": CLIP_SOURCE,
        "clip_preprocessor": CLIP_SOURCE,
    },
    auto_download=False,
) as session:
    # vibe can now directly accept file paths (string or Path)
    inference_result = session.infer("example/example.jpg")

    # Access the result using the .first() helper
    result = inference_result.first()

    # ScoreResult includes raw score and normalized_score out of the box
    print("Raw Score:", result.score)
    print("Normalized:", result.normalized_score)
