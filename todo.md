# EE

- log device and precision that actually ended up being used
  - variant id etc too i guess

- give result transforms properties similar to PluginOptionSpec, so UI can auto generate components based on that?

- think about making ModelIdentity.description required

- look into jtp/hydra validation data and see if its usable with tag level thresholds
  - Hydra has the validation data embedded into weights as "validation" tensor, jtp 3 uses separate file

- move auto_download to config instead of load()?

- add some kind of pre-inference check to transforms to ensure everything is fine (eg files are available and/or download-able) so they dont error only when they are run which is after inference of a batch has already finished, wasted time

## Todo

- think about adding HF token param, technically env var/hf cli login will do but it seems like its common practice to have token param
  - related: global env vars or similar file so users can set specific settings that arent meant to be changed but the chance is there it can resolve an issue or similar (debugging etc)
    - to enable true batching for CPU
    - reduce prefetch batch limit from hardcoded 8 to lower

- Availability checks
  - If model_id set needs is available in HF cache or in set path, and what files are missing, if any, and return if none is available
  - also have some pre-inference check for result processors, or just CharacterIPMapping
    - if auto_download=False then the RP will error

- log that selection != CPU but only CPU EP available and suggest user to check install
  - log if selected backend doesnt exist
  - log more things similar to above

- Make precision selection more clearer, upd: need to check again

- Add (onnx) openvino int8 weights, use pixai tagger hf space for reference

- Model IDs, robustness, uniqueness
  - ~~Aliases too?~~ aliases removed, model_id should be unique strings, at best we can have "Possible names" or similar to aid in search mechanisms
  - Possibility to set model_id in vibe.load() to HF repo?

- Finish documentation/doc strings

- (V2) add possibility to use embedding output
  - add onnx tensor output mapping and/or have it be the "output capabilities" (eg this onnx model supports embedding, prediciton, logits output and whatnot etc etc)

- (linked) add to model spec to show which output is what thing (logit, predictions, embeddings)
  - check again if theres possibility to overwrite spec stuff so it can be used but with some stuff customized, so its possible to force usage of specific output index

- figure out a more concrete way of how image size is dealt with
  - animetimm models use image size from timm config files
  - SW wd taggers can use a timm config.json too (no preprocess json but normal config includes size)
  - how to deal with dynamic image size input?
    - Nothing for this yet / JTP3/hydra has extra setting seqlen but that is kinda specific to the model when dynamic image size input is quite general, eg taggerine should support similar option, or i give taggerine a custom config setting too

## Things that need real world testing

- hf_revision param usage
- generic-timm* model IDs need testing, never used them
