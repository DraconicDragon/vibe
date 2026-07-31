# new new todo huehuehuehue

- add some kind of pre-inference check to transforms to ensure everything is fine (eg files are available and/or download-able) so they dont error only when they are run which is after inference of a batch has already finished, wasted time

- [x] add validation of some kind to ensure metadata is same as what gets actually outputted, or just error if different
  - validation added is simple metadata validation at runtime
    - add validation pre-inference (unit tests?) and whole pipeline validation (plugin validator module, possibly with unit tests)

## Todo

- think about adding HF token param, technically env var/hf cli login will do but it seems like its common practice to have token param
  - related: global env vars or similar file so users can set specific settings that arent meant to be changed but the chance is there it can resolve an issue or similar (debugging etc)
    - to enable true batching for CPU
    - reduce prefetch batch limit from hardcoded 8 to lower

- ensure that heavy dependencies (torch, onnxruntime, transformers) are strictly imported inside load_ancillary or preprocess / postprocess

- Availability checks
  - If model_id set needs is available in HF cache or in set path, and what files are missing, if any, and return if none is available
  - also have some pre-inference check for result processors, or just CharacterIPMapping
    - if auto_download=False then the RP will error

- log that selection != CPU but only CPU EP available and suggest user to check install
- log if selected backend doesnt exist
- log more things similar to above

- tend to generic-timm* ModelPlugins

- refactor precision to weight_precision
  - Can have weight in fp16/bf16 and compute in fp16/bf16 or fp32 too, scores change by a very tiny a mount
    - for cpu compute is usually fp32 only, but weights can be fp16, are auto upcasted to fp32 at inference without any notable memory increase, may decrease in speed by a very tiny bit for upcasting
  - compute_precision/dtype is different story, most of the time compute runs in fp32 anyway, but lower weight dtype helps with memory usage

- Make precision selection more clearer, upd: need to check again
- Add (onnx) openvino int8 weights, use pixai tagger hf space for reference

- Model IDs, robustness, uniqueness
  - ~~Aliases too?~~ aliases remove, model_id should be unique strings, at best we can have "Possible names" or similar to aid in search mechanisms
  - Possibility to set model_id in vibe.load() to HF repo?

- Finish documentation/doc strings

- torch xpu, should be single line check to add support for that

- (V2) add possibility to use embedding output
  - add onnx tensor output mapping and/or have it be the "output capabilities" (eg this onnx model supports embedding, prediciton, logits output and whatnot etc etc)
- (linked) add to model spec to show which output is what thing (logit, predictions, embeddings)
  - check again if theres possibility to overwrite spec stuff so it can be used but with some stuff customized, so its possible to force usage of specific output index

- (V2?) model-specific input values
  - have a some way (function?) that lets user choose settings for that specific model
    - see JTP 3 / Hydra 3.5, its possible to control sequence length, and add loras technically

- figure out a more concrete way of how image size is dealt with
  - animetimm models use image size from timm config files
  - SW wd taggers can use a timm config.json too (no preprocess json though) but config includes sizes
  - how to deal with dynamic image size input?
    - Nothing for this yet

## Things that need real world testing

- hf_revision param usage
- generic-timm* model IDs need testing, never used them
