from PIL import Image

import autotagger

MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"

# Using 'with' is optional but calls session.close() automatically to free resources when done.
with autotagger.load("wd-eva02-large", source=MODEL_SOURCE) as session:
    result = session.infer(Image.open("example/example.jpg")).first()

    # Result already sorted by score (high to low)
    score_dict = result.as_score_dict()

    # only top 10 tags by score
    top_10_scores = list(score_dict.items())[:10]

    for tag, score in top_10_scores:
        # Print the tag with a score rounded to 3 decimal places
        print(f"  {tag}: {score:.3f}")
