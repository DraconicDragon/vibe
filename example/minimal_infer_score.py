import logging

from PIL import Image

import vibe

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("vibe").setLevel(logging.WARNING)


with vibe.load(
    "waifu-scorer-v3",
    # auto_download=False,
) as session:
    result = session.infer(
        Image.open("example/example.jpg"),
    ).first()

    print(f"score: {result.score:.3f}")
