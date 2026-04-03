import autotagger

MODEL_SOURCE = "local:/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3"

image_paths = [
    "example/example.jpg",
    "example/example.jxl",
]

# Using 'with' is optional but calls session.close() automatically to free resources when done.
with autotagger.load("wd-eva02-large", source=MODEL_SOURCE, auto_download=False) as session:
    # Prepare inputs as (input, reference) pairs
    inputs = [(path, path) for path in image_paths]

    # Run inference in batches (auto handles file types like jpg/jxl)
    batch = session.infer(inputs, batch_size=min(8, len(inputs)), batch_method="auto")

    for item in batch:
        print(f"\nImage #{item.index} ({item.input_ref}):")

        # Result already sorted by score (high to low)
        score_dict = item.result.as_score_dict()

        # only top 10 tags by score
        top_10_scores = list(score_dict.items())[:10]

        for tag, score in top_10_scores:
            # Print the tag with a score rounded to 3 decimal places
            print(f"  {tag}: {score:.3f}")
