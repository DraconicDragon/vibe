from PIL import Image

import vibe

MODEL_SOURCE = "local:/home/drac/dev/models/seperate/wd-eva02-large-tagger-v3/"

# Using 'with' is optional but calls session.close() automatically to free resources when done.
with vibe.load("wd-eva02-large-v3", source=MODEL_SOURCE, backend="onnx") as session:
    result = session.infer(Image.open("example/example.jpg")).first()

    # print(result.to_dict())
    # print()

    if vibe.is_tag_result(result):
        # print(result.tags)
        #print(result.tags[:100])

        # Result already sorted by score (high to low)
        score_dict = result.as_score_dict()

        # only top 10 tags by score
        top_10_scores = list(score_dict.items())[:20]

        for tag, score in top_10_scores:
            # Print the tag with a score rounded to 3 decimal places
            print(f"  {tag}: {score:.5f}")
