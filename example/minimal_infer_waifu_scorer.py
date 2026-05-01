import logging

from PIL import Image

import vibe

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("vibe").setLevel(logging.WARNING)

MLP_SOURCE = "/mnt/T7/Projects/GitHub/vibe/models/waifu-scorer-v3"
CLIP_SOURCE = "/mnt/T7/Projects/GitHub/vibe/models/clip-vit-large-patch14"

with vibe.load(
    "waifu-scorer-v3",
    source_map={
        "Eugeoter/waifu-scorer-v3": MLP_SOURCE,
        "openai/clip-vit-large-patch14": CLIP_SOURCE,
    },
    auto_download=False,
) as session:
    inference_result = session.infer(
        Image.open("example/example.jpg"),
        result_processors=[vibe.NormalizedScore()],
    )

    print(inference_result.items[0].result.score)
    print("Normalized:", inference_result.items[0].result.normalized_score)
