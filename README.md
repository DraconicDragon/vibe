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

TBD

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

Unfinished/TBD

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

TBD

### Documentation

TBD

#### AI-generated code disclaimer

<details><summary>Expand to read</summary>

This project contains a great deal of AI generated code.  
Below is a list of models I've used.

- Gemini 3.5 flash/3.6 flash/3.1 Pro - Most of the actual code writing. Done under my supervision, however good that may have been.
- Claude Sonnet 4.6/5.0 - planning, review
- GPT 5.2/5.3 Codex/5.5/5.6 Terra/5.6 Luna - rarer, larger one-off edits, planning, review
- Qwen 3.7 Plus/3.7 Max/3.8 Max, GLM 4.7/5.1 - miscellaneous things, questions
- Gemini 3.1 flash, Claude Haiku 4.5, GPT-5.4 mini, Grok Code Fast 1 - smaller inline edits

</details>
