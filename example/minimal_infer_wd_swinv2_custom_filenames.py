from PIL import Image

import vibe

# Example: local folder has custom names instead of model.onnx / selected_tags.csv
MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-swinv2-tagger-v3-custom"
FILE_NAME_MAP = {
    "model.onnx": "wd_swinv2_v3_fp16.onnx",
    "selected_tags.csv": "selected_tags_custom.csv",
}

with vibe.load(
    "wd-swinv2-v3",
    source=MODEL_SOURCE,
    backend="onnx",
    auto_download=False,
    file_name_map=FILE_NAME_MAP,
) as session:
    result = session.infer(Image.open("example/example.jpg")).first()

    for tag, score in list(result.as_score_dict().items())[:10]:
        print(f"  {tag}: {score:.3f}")
