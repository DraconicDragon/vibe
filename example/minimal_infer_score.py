import logging

from PIL import Image

import vibe
from vibe.results import is_multi_score_result, is_score_result

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("vibe").setLevel(logging.WARNING)

MODEL_SOURCE = "/mnt/T7/Projects/GitHub/vibe/models/anime_aesthetic/swinv2pv3_v0_448_ls0.2_x"
USE_SCORE_RESULT_PROCESSOR =  True

result_processors = []
if USE_SCORE_RESULT_PROCESSOR:
    from vibe.result_processors import MultiScoreToScore

    result_processors.append(MultiScoreToScore(use_samples_percentile=False))

with vibe.load(
    "dghs-aes-swinv2pv3-ls0.2-x",
    source=MODEL_SOURCE,
) as session:
    result = session.infer(
        Image.open("example/example.jpg"),
        result_processors=result_processors,
    ).first()

    if is_multi_score_result(result):
        print(result.scores)
        print(result.metrics)
    elif is_score_result(result):
        print(result.score)
    else:
        print(result)
