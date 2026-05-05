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
USE_SCORE_RESULT_PROCESSOR = True

result_processors = []
if USE_SCORE_RESULT_PROCESSOR:
    from vibe.result_processors import MultiScoreToScore, NormalizedScore

    # If you want multiscore result as the model outputs it, comment the line below and uncomment the line below that
    result_processors.append(NormalizedScore(use_samples_percentile=False))
    # result_processors.append(MultiScoreToScore(use_samples_percentile=True))


with vibe.load(
    "dghs-aes-swinv2pv3-ls0.2-x",
    source=MODEL_SOURCE,
) as session:
    result = session.infer(
        Image.open("examples/example.jpg"),
        result_processors=result_processors,
    ).first()

    if is_multi_score_result(result):
        print("Aesthetic Scores:")
        label_scores = result.as_label_score_dict()
        if result.label_order is not None:
            for label in result.label_order:
                if label in label_scores:
                    print(f"  {label}: {label_scores[label]:.4f}")
        else:
            for label, score in label_scores.items():
                print(f"  {label}: {score:.4f}")
        print(f"Normalized Score: {result.normalized_score:.4f}")
    elif is_score_result(result):
        print(f"Score: {result.score:.4f}")
        print(f"Normalized Score: {result.normalized_score:.4f}")
    else:
        print(result)
