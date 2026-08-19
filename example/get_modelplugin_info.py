import vibe

model_id = "taggerine"
info = vibe.describe(model_id)

print(info)
print(f"\nFiles & Variants for {model_id} (Family: {info.family_name}):")

# Iterate through model variants (each variant represents a backend target)
for variant in info.variants:
    variant_label = f"[{variant.backend.value}]"
    if variant.variant_id:
        variant_label += f" (variant: {variant.variant_id})"
    
    print(f"\nBackend Variant: {variant_label}")
    for artifact in variant.artifacts:
        req_str = "required" if artifact.required else "optional"
        subdir_str = f" | dir: {artifact.hf_subdir}" if artifact.hf_subdir else ""
        print(f"  • {artifact.name:<25} ({artifact.role.value}, {req_str}{subdir_str})")