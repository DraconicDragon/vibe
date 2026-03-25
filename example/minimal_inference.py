"""
Minimal example: load an image and tag it with a model.
"""

from PIL import Image

import autotagger
from autotagger import is_multi_score_result, is_score_result, is_tag_result

MODEL_NAME = "wd-eva02-large"

    # todo: add ability to pass local model folder(s) to pass for checking and availability stuff and whatnot (needs more info)
# todo: also HF folder name detection if not already there so that passed on folder is seen as HF location and will check repo id

# Load the model (downloads from HuggingFace automatically on first run)
session = autotagger.load(MODEL_NAME)

# Load an image
image = Image.open("example.jpg").convert("RGB")

# Run inference
result = session.infer(image)

# Process result based on its type using TypeGuard narrowing
if is_tag_result(result):
    print("Tags:", result.tag_names())
    print("\nScores:")
    for tag, score in result.as_score_dict().items():
        print(f"  {tag}: {score:.3f}")
elif is_score_result(result):
    print(f"{result.label}: {result.score:.3f} (range {result.score_min}..{result.score_max})")
elif is_multi_score_result(result):
    for name, value in result.scores.items():
        print(f"  {name}: {value:.3f}")
else:
    # Fallback for forward compatibility
    print(result.to_dict())

# Clean up
session.close()
