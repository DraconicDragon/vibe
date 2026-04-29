import logging

from PIL import Image

import vibe

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("vibe").setLevel(logging.WARNING)

MODEL_SOURCE = "/mnt/T7/Projects/GitHub/vibe/models/deepghs/anime_aesthetic/swinv2pv3_v0_448_ls0.2_x"

with vibe.load(
    "dghs-aes-swinv2pv3-448-ls0.2-x",
    source=MODEL_SOURCE,
) as session:
    result = session.infer(
        Image.open("example/example.jpg"),
    ).first()

    print(result.scores)
    print(result.metrics)