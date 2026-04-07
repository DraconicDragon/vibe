import vibe

model_id = "wd-eva02-large-v3"
info = vibe.describe(model_id)
print(info)
print(f"Files for {model_id}:")
for f in info.required_files:
    # Show backends in brackets, or [all] if empty
    backends = f"[{', '.join(b.value for b in f.backends)}]" if f.backends else "[any]"
    print(f"  • {f.name:<20} {backends:<15} ({f.role.value})")
