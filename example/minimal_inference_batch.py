import autotagger

MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"
image_paths = [
    "example/example.jpg",
    "example/example.jxl",
]

with autotagger.load("wd-eva02-large", source=MODEL_SOURCE, auto_download=False) as session:
    inputs = [(path, path) for path in image_paths]
    batch = session.infer(inputs, batch_size=min(8, len(inputs)), batch_method="auto")

    for item in batch:
        print(f"\nImage #{item.index} ({item.input_ref}):")
        top_10 = list(item.result.as_score_dict().items())[:10]
        for tag, score in top_10:
            print(f"  {tag}: {score:.3f}")
