# Vibe

Vibe, or: Vision transformer Inference Backend (very creative name I know, thank you), is a library to quickly get started with end to end inference using (mostly) my favourite (small) vision transformers - like wd taggers, aesthetic scorers, and other classifiers.  

## TLDR

Personal, vibecoded project. Messy, rough, but works on my machine™️ for my purposes®️
Easy to get started with (probably); Check [Quick Start section](#quick-start-examples), [examples folder](./example/) in this repo and/or [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md) if you want to get started using vibe without reading the garbage below. Don't forget to [install](#installation) onnx and/or pytorch.

## Info

It's mostly a personal project and it is vibecoded, if the name didn't tell you that already.  
I encourage you to check out [deepghs' imgutils](https://github.com/deepghs/imgutils) instead before using vibe, since imgutils has more complete support of various tagging models and then some as well as being more mature and likely more robust. I mostly just wanted my own little library that I can easily use that comes with things I want from it.  
My plan is to support models that I will likely find usage in; which may or may not be models that imgutils already supports.

I may be open to feature requests but I would be surprised that 1. You found this project, and 2. You are (willing to) using it and/or want something changed/new added

### Things that work, or "Features"

TODO  

For a list of supported models see: [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md)

- Supports either ONNX or PyTorch models, or both! At least one backend is a required
  - useful to save space I guess; PyTorch models may perform better at larger batches on GPU(?)
- Simple usage I guess, you mainly only need to vibe.load() a session and then session.infer()
- Batching for better performance (throughput) on GPU (and CPU if explicitly passed)
  - I'm mentioning this because it seems like a lot of projects only do one image at a time when the models actually support batch inputs which are a bit faster than one image per inference call on GPU (or CPUs with stupid amount of cores I believe). On normal CPUs batch tensor input seems to be slower than one image at a time so vibe will default to "true" batch for GPU and "sequential" if using CPU, unless you explicitly pass `batch_method="true"`
- Batched input will load the next chunk of images on demand as inference runs to improve throughput.
- Local folder and HuggingFace support - Something I personally really wanted because hate reliance on HF cache (or it's folder structure) and/or HF download since I have some models downloaded separately.
  - Using vibe.load() you can specify `source=` by using either a HF repo id or a path to a local folder
    - You can also prefix the string with `local:` or `hf:` to be explicit about where you want vibe to look for the files
      - If you want to specifically have *only* HF Cache looked at, use `hf:` prefix with `auto_download=False` in `vibe.load()`
  - You can set `auto_download=False` if you don't want to have ANY missing file downloaded from HF; It will error if a requiredfile is missing and can't be downloaded.
  - For local folder support it's expected to have the required files in the specified folder
    - You can check for which files are required by using `vibe.describe(model_id)`; check [get_modelplugin_info.py](./example/get_modelplugin_info.py) for more info
  - You can also specify custom file mapping (TODO: more info needed)

- Sync (`vibe.infer()`) and async (`vibe.infer_async()`) support, both cancel-able
  - Cancel happens when possible, meaning after current image load or inference chunk is done
  - sync `infer()` requires separate thread to call cancel
  - Cancelling returns `InferenceCancelled` error
    - In the case of sync `infer()` call it will allow you to control what to do upon cancelling - either return just the `InferenceCancelled` error or return already processed results - for `infer_async()` you can already get the result as they are processed anyway

**Result processing:**

Result Processors are a set of classes that can be supplied to the `result_processors` arg in `infer()`/`infer_async()`.  
As the name suggest, if these classes are supplied they will process the result before returning it.

ModelPlugins specify which result processors are supported but you are free to add any you want - A warning will be logged if you try to use an unsupported one, but vibe will attempt to apply it anyway; May or may not work. Can be useful if you want to make/use your own result processor class

The CleanTags result processor will always run last (subject to change; this is mostly so other result processors don't use the output from CleanTags which would cause issues)

TODO: info on how to get result processor + param info

<details><summary>Less notable stuff</summary>

- JPEG-XL support through `pillow-jxl-plugin`
- Memory tracking - have it added because, but didn't do much with it except for testing memory usage to find potential leaks
- Type-safe results (TagResult, ScoreResults, MultiScoreResult) with TypeGuard support for type-safe integration in IDEs so type checker doesnt whine about it

</details>

<details><summary>Things I don't plan to add that imgutils already has</summary>

- Gradio quick demo; out of scope
- old models like mldanbooru, deepdanbooru, cdc upscaler, etc - unless if requested, then I might bother
- metrics module, eg ccip/lpips metrics, clustering - not sure if useful
- resource module to get (background) image files; out of scope
- preprocess module (probably, but I don't see it being useful for me)
- sd module; out of scope - contains utilities for sd webui/NAI metadata
- metadata module; out of scope - similar but more general for EXIF, LSB (Least Significant Bit) etc
- data module; out of scope - probably
  - I thought about adding something like data.url submodule but decided against it since managing image downloads and all is imo out of scope too, but I can recommend `curl_cffi` package for easily getting images from various sources with small hassle; at least that was my experience with it
- ascii module; out of scope

</details>

---

## Installation

By default (if you just install through `pip install git+https://github.com/DraconicDragon/vibe`), vibe does not come with a backend runtime. You need at least one of:

- PyTorch
- ONNX

Both backends can coexist freely; e.g. ONNX CPU + PyTorch CUDA is fine.  
If you want only one, can't decide and you are fine with either, I suggest going with PyTorch. There are more models out there that come in PyTorch compatible format than ONNX, and it's possible it'll be less hassle

<details><summary>CPU</summary>

If you plan to use GPU, you can skip this.

**PyTorch:**

```bash
pip install "git+https://github.com/DraconicDragon/vibe[torch-cpu]"
```

**ONNX:**

```bash
pip install "git+https://github.com/DraconicDragon/vibe[onnx-cpu]"
```

**Both:**

```bash
pip install "git+https://github.com/DraconicDragon/vibe[torch-cpu,onnx-cpu]"
```

</details>

<details><summary>NVIDIA GPU</summary>

<details><summary>PyTorch + CUDA</summary>

The pyproject.toml file does not include an extra optional dependency for torch+CUDA.

You can replace cu128 in the --index-url to whatever version you like and that is also supported.  
RTX 50 series / Blackwell requires at least cu128 (I think?) and torch 2.7.x seems to be the earliest pytorch version to come with cuda 12.8 builds

```bash
pip install git+https://github.com/DraconicDragon/vibe \
  "torch>=2.7.1" \
  "safetensors>=0.6.2" \
  "timm>=1.0.22" \
  "transformers>=5.0.0" \
  "einops" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

> **Note:** For Maxwell/Pascal GPUs you made need to replace cu128 with cu126 or cu124

</details>

<details><summary>ONNX + CUDA</summary>

> **Note:** Don't install more than one `onnxruntime` variant (`onnxruntime`, `onnxruntime-gpu`,
> `onnxruntime-rocm`, etc.) in the same environment. They overwrite each other's entry points and the conflict won't be reported as an error and will just silently break
> (for example, if you had onnxruntime-gpu installed at first, then install onnxruntime - you will only have access to the execution providers from onnxruntime and need to uninstall and install onnxruntime-gpu again to be able to access CUDA EP and TensorRT EP)
> If you encounter any weirdness, try to uninstall any onnxruntime package you have and then just install the one you want to have

**Windows**:

```bash
pip install "git+https://github.com/DraconicDragon/vibe[onnx-cuda]"
```

**Linux + System level CUDA/cuDNN / PyTorch + CUDA**:

This is a bit weird for me but as far as I understood CUDA in relation to linux and onnxruntime:  
If torch+cuda is installed it should already come with the required nvidia packages and this lib will try to search for those in the same environemnt (works if torch's cuda version is the one that `onnxruntime` expects, usually cu12x). Otherwise you will need to install CUDA/cuDNN libraries from your system package manager, *or* through pip packages - info for that further below.

```bash
pip install "git+https://github.com/DraconicDragon/vibe[onnx-cuda]"
```

> ~~**Note:** If your CUDA/cuDNN is version 13.x, you need onnxruntime-gpu to be v1.24.1 or higher - pyproject.toml pins the version to '>=1.17.3' so it should automatically install 1.24.1 or higher, but to be sure you can do `pip install "git+https://github.com/DraconicDragon/vibe[onnx-cuda-cu13]"`~~

**Linux, no torch/system CUDA**:

Pulls CUDA libraries from PyPI  
You will likely need this if you are on linux, and do not have torch+cuda installed or CUDA/cuDNN installed through your package manager.

```bash
pip install "git+https://github.com/DraconicDragon/vibe[onnx-cuda-linux-bundled-cu12]"
```

> ~~**Note:** If you want to use Cuda 13.x, do `pip install "git+https://github.com/DraconicDragon/vibe[onnx-cuda-linux-bundled-cu13]"`~~

</details>

</details>

<details><summary>AMD GPU (ignore)</summary>

I don't have an AMD gpu to test. So there's no instructions here. But it might basically be the same as installing torch and onnxruntime with rocm for other projects... in theory.

</details>

<details><summary>Intel GPU (ignore)</summary>

I don't have an Intel gpu to test. So there's no instructions here. But it may be just installing torch xpu and onnxruntime with... onednn (or something?) for other projects... in theory.
Theres official torch xpu packages but i dont think vibe currently takes into account xpu as device so there might need to be 1-2 lines of code changed

</details>

### Quick start examples

Single image, minimal inference using wd swinv2 tagger v3 from SmilingWolf

```py
from PIL import Image

import vibe

# Using 'with' is optional but calls session.close() automatically to free resources when done.
# You can also provide source="/path/to/model_folder/" or source="hf_repo/id".
with vibe.load("wd-swinv2-v3") as session:
    # infer() supports image path, PIL object, numpy image or a list with any of those types.
    # optionally you can also pass a list of (image_or_path, ref) tuples where ref is a unique string 
    # so that you can refer to the image easier in the output. 
    # By default a ref is created either way and just uses the index of the image/object in the list.
    result = session.infer(Image.open("example/example.jpg")).first()

    # Result already sorted by score (high to low)
    score_dict = result.as_score_dict()

    # only top 10 tags by score
    top_10_scores = list(score_dict.items())[:10]

    for tag, score in top_10_scores:
        # Print the tag with a score rounded to 3 decimal places
        print(f"  {tag}: {score:.3f}")
```

- [minimal_infer.py](/example/minimal_infer.py) - almost same thing as example codeblock above
- [minimal_infer_batch.py](/example/minimal_infer_batch.py) - same thing as above but with batch image input
- [infer_batch_async_with_cancel.py](/example/infer_batch_async_with_cancel.py) - async batch inference example with cancel support (press c to cancel) and logging setup
- [infer_multi_model.py](/example/infer_multi_model.py) -  async inference with 2 models doing batch thing at same time
- [vibe_utility_print](/example/vibe_utility_print.py) - barebones, might be useful not sure, prints out some info like available devices, runtime package install info
- You can find more in the [examples folder](/example/)

### Documentation

TODO, theres some amount of docstrings available, but no auto generated doc page or similar

#### AI-generated code disclaimer

<details><summary>Expand to read</summary>

This project contains a good deal of AI generated code.  
Below is a list of models I've used and roughly the main purposes I used them for

- Claude Sonnet 4.6; great helper in solidifying my initial intention and vision for the structure of the project and made base skeleton code for project to get started with; also quite pleasant to talk with
- GPT 5.2/5.3 Codex; Large, complex edits, bigger feature additions, more complex questions, most tests
- Claude Haiku 4.5; Quite helpful for many, many small edits and some simpler questions - feels like pretty good balance of cost, reliability, speed etc
  - I prefer GPT-5.4 mini with thinking disabled a bit more since it released
- Gemini 3.1 flash & Grok Code Fast 1; mostly for inline edits and smaller edits - very direct with how it does things (usually)

</details>
