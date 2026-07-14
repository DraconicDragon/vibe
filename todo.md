# new todo

- check session factory build_session, do the args need to be defaulted to something or nah? the only true places its being used, load/load_custom, already set all things
  - also session.py __init__
- change default backend selection from onnx to pytorch?

- pillow jxl/heif availability check is in both session.py and image_loading.py

## Todo

- character ip mapping, change default: put resolved ip mapping in copyright category, and give it confidence score that is same as gotten character
  - also look at logic again to see how it does things

- function based result processors? this way docstrings

- Availability checks
  - If model_id set needs is available in HF cache or in set path, and what files are missing, if any, and return if none is available
  - also have some pre-inference check for result processors, or just CharacterIPMapping
    - if auto_download=False then the RP will error

- log that selection != CPU but only CPU EP available and suggest user to check install
- log if selected backend doesnt exist
- log more things similar to above

<!-- - make generic model class, eg for timm, like dghs imgutils.generic, should be possible assuming timm config etc is all there, wd taggers and animetimm taggers have it at least 
  - or AT LEAST something so hf repo or local folder can be automatically resolved to model id so person doesnt need to know about model IDs if using HF repo id or hf repo id as local folder name-->

- refactor precision to weight_precision
  - Can have weight in fp16/bf16 and compute in fp16/bf16 or fp32 too, scores change by a very tiny a mount
    - for cpu compute is usually fp32 only, but weights can be fp16, are auto upcasted to fp32 at inference without any notable memory increase, may decrease in speed by a very tiny bit for upcasting
  - compute_precision/dtype is different story, most of the time compute runs in fp32 anyway, but lower weight dtype helps with memory usage

- Make precision selection more clearer
- Add (onnx) openvino int8 weights, use pixai tagger hf space for reference

- Model IDs/names should be in some concrete standardized-ish form
  - Aliases too?
  - I do want it to be possible for model id to just be HF repo name too, which can be automatically an alias i guess
    - Find out if this would cause any issues, probably not

- Finish documentation/doc strings

- torch xpu, should be single line check to add support for that

- (V2) add possibility to use embedding output
  - add onnx tensor output mapping and/or have it be the "output capabilities" (eg this onnx model supports embedding, prediciton, logits output and whatnot etc etc)
- (linked) add to model spec to show which output is what thing (logit, predictions, embeddings)
  - check again if theres possibility to overwrite spec stuff so it can be used but with some stuff customized, so its possible to force usage of specific output index

- (V2?) per model values
  - result processors can use it
  - have a some way (function?) that lets user choose settings for that specific model
    - see JTP 3 / Hydra 3.5

- rename precision to weight_precision
  - and add compute_precision (at least for torch)

- figure out a more concrete way of how image size is dealt with
  - animetimm models use image size from timm config files
  - SW wd taggers can use a timm config.json too (no preprocess json though) but config includes sizes
  - how to deal with dynamic image size input?
    - Nothing for this yet

- think about adding HF token param, technically env var/hf cli login will do but it seems like its common practice to have token param

## Things that need real world testing

- Precision string input
- hf_revision param usage
- generic-timm* model IDs need testing, never used them
