"""
More advanced simple example: list available models and auto-load the first one.
"""

from typing import Any

import vibe


def list_available_models() -> list[dict[str, Any]]:
    """
    Return a list of all available models with their aliases.

    Each entry is a dict with:
      - model_id: canonical model identifier
      - aliases: list of alternative names for the model
      - display_name: human-readable name
      - description: what the model does
    """
    models = []
    for model_id in vibe.list_models():
        info = vibe.describe(model_id)
        models.append(
            {
                "model_id": info.model_id,
                "aliases": info.aliases,
                "display_name": info.display_name,
                "description": info.description,
            }
        )
    return models


# List all available models
available_models = list_available_models()

print("Available models:")
for i, model_info in enumerate(available_models):
    aliases_str = f" (aliases: {', '.join(model_info['aliases'])})" if model_info["aliases"] else ""
    print(f"  [{i}] {model_info['model_id']}{aliases_str}")
    print(f"      {model_info['description']}")

# Use the first model
chosen_index = 0
chosen_model_id = available_models[chosen_index]["model_id"]
chosen_aliases = available_models[chosen_index]["aliases"]

print(f"\nUsing model: {chosen_model_id}")
if chosen_aliases:
    print(f"Aliases: {', '.join(chosen_aliases)}")

# ============================================================================
# Everything below is commented out—uncomment to run inference
# ============================================================================

# session = vibe.load(chosen_model_id)
#
# # Load an image
# from PIL import Image
# image = Image.open("example.jpg").convert("RGB")
#
# # Run inference
# result = session.infer(image)
#
# # Process result based on its type using TypeGuard narrowing
# if vibe.is_tag_result(result):
#     print("Tags:", result.tag_names())
#     print("\nScores:")
#     for tag, score in result.as_score_dict().items():
#         print(f"  {tag}: {score:.3f}")
# elif vibe.is_score_result(result):
#     print(f"{result.label}: {result.score:.3f} (range {result.score_min}..{result.score_max})")
# elif vibe.is_multi_score_result(result):
#     for name, value in result.scores.items():
#         print(f"  {name}: {value:.3f}")
# else:
#     # Fallback for forward compatibility
#     print(result.to_dict())
#
# # Clean up
# session.close()
