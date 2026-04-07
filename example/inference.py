from PIL import Image

import vibe
from vibe import is_multi_score_result, is_score_result, is_tag_result

MODEL_NAME = "wd-eva02-large-v3"
MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"

IMAGE_LIST = [
    "example/example.jpg",
    "example/example.jxl",
]

with vibe.load(MODEL_NAME, source=MODEL_SOURCE, auto_download=False) as session:
    # Supported input forms:
    # session.infer(image)
    # session.infer("path/to/image.jpg")
    # session.infer([img1, img2, img3])
    # session.infer(["a.jpg", "b.jpg"])
    # session.infer([(img1, "a"), (img2, "b")])

    image = Image.open(IMAGE_LIST[0])
    batch = session.infer(image)

    # InferenceResult is the same shape for single and batch input.
    # .first() is a convenience for the single-image case.
    # Iterate .items for batch results.
    single_result = batch.first()
    if is_tag_result(single_result):
        top_scores = sorted(single_result.as_score_dict().items(), key=lambda x: x[1], reverse=True)[:10]
        print("Single input top tags:")
        for tag, score in top_scores:
            print(f"  {tag}: {score:.3f}")

    # create batch_inputs from IMAGE_LIST, using filename as ref
    batch_inputs = [(path, path) for path in IMAGE_LIST]
    batch_result = session.infer(batch_inputs, batch_size=2, batch_method="auto")

    for item in batch_result.items:
        print(f"\nItem index={item.index} ref={item.input_ref!r}")
        result = item.result
        if is_tag_result(result):
            top_scores = sorted(result.as_score_dict().items(), key=lambda x: x[1], reverse=True)[:10]
            for tag, score in top_scores:
                print(f"  {tag}: {score:.3f}")
        elif is_score_result(result):
            print(f"  {result.label}: {result.score:.3f} (range {result.score_min}..{result.score_max})")
        elif is_multi_score_result(result):
            for name, value in result.scores.items():
                print(f"  {name}: {value:.3f}")
        else:
            print(f"  {result.to_dict()}")
