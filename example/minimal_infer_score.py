import logging

import vibe
from vibe.results import is_multi_score_result, is_score_result

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

MODEL_SOURCE = "/home/drac/dev/models/seperate/anime_aesthetic/swinv2pv3_v0_448_ls0.2_x/"

with vibe.load(
    "dghs-aes-swinv2pv3-ls0.2-x",
    source=MODEL_SOURCE,
) as session:
    result = session.infer("/mnt/P5P/Hydrus/hydrus-source/db/client_files/f9d/c/9dcd513602d7b162e5778c40889a1ca32092954b266ec604aaf6a8a8fc69063b.png").first()

    if is_multi_score_result(result):
        print("Aesthetic Scores:")
        for entry in result.entries:
            print(f"  {entry.label}: {entry.score:.4f} (normalized: {entry.normalized_score:.4f})")

        print(f"Overall Normalized Score: {result.normalized_score:.4f}")

        if "percentile" in result.extras:
            print(f"Calibrated Percentile: {result.extras['percentile'] * 100:.2f}%")

    elif is_score_result(result):
        print(f"Score ({result.label}): {result.score:.4f}")
        print(f"Normalized Score: {result.normalized_score:.4f}")

    else:
        print(result.to_dict())
