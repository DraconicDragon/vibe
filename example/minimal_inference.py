"""
Minimal example; load an image and tag it with a model.
"""

from PIL import Image

import autotagger
from autotagger import is_multi_score_result, is_score_result, is_tag_result

MODEL_NAME = "wd-eva02-large"
# can omit local: prefix, or use hf or hf_cache, omitting will try local -> hf cache -> hf repo auto download if repo ID
MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"

# Load model
# NOTE: if 'source' is not provided, by default it will just look for the HF repo id
# defined in the ModelPlugin subclass in HF cache first,
# then auto download so long as auto_download != False
session = autotagger.load(MODEL_NAME, source=MODEL_SOURCE, auto_download=False)

# Load an image, is turned to RGB internally based on model implementation (which is most)
image = Image.open("example/example.jpg")

# Run inference
result = session.infer(image)

# Process result based on its type using TypeGuard narrowing
if is_tag_result(result):
    print("Top 15 Tags with Scores:")
    top_scores = sorted(result.as_score_dict().items(), key=lambda x: x[1], reverse=True)[:15]
    for tag, score in top_scores:
        print(f"  {tag}: {score:.3f}")
elif is_score_result(result):
    print(f"{result.label}: {result.score:.3f} (range {result.score_min}..{result.score_max})")
elif is_multi_score_result(result):
    for name, value in result.scores.items():
        print(f"  {name}: {value:.3f}")
else:
    # Fallback for forward compatibility
    print(str(result.to_dict())[:700])

# Clean up
session.close()
