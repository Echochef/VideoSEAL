<h1 align="center">🎬 VideoSEAL</h1>

<h3 align="center">Separating Planning from Answer Authority for Agentic Long Video Understanding</h3>

<p align="center">
  <a href="https://huggingface.co/CewEhao/VideoSEAL_8B">
    <img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-VideoSEAL__8B-yellow">
  </a>
  <a href="#-citation">
    <img alt="ICML 2026" src="https://img.shields.io/badge/ICML-2026-blue">
  </a>
  <a href="https://github.com/Echochef/VideoSeal">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-black?logo=github">
  </a>
</p>

<p align="center">
  Official implementation of <b>VideoSEAL</b>, an agentic framework for long-video understanding that
  decouples <i>planning</i> (which evidence to gather) from <i>answer authority</i> (what the answer is).
</p>

---

## 📰 News

- **`2026-05-01`** &nbsp;🎉 Our paper **VideoSEAL** has been **accepted to ICML 2026**!
- **`2026-01-27`** &nbsp;🚀 Initial release of code and the [`VideoSEAL-8B`](https://huggingface.co/CewEhao/VideoSEAL_8B) checkpoint.

## 📖 Overview

VideoSEAL is an agentic pipeline for long-video question answering. It separates the *planner* role
(deciding what to look at) from the *answerer* role (judging the evidence), and trains both with GRPO
over a tool-augmented rollout.

The repository covers the full stack:

- 🧱 **Offline build** — convert raw videos into a unified semantic index under `indexes/semantic/<video_id>/`
- 🛠️ **Tool-using agent** — OCR-subtitle search, clip captioning, and visual inspection
- 🏋️ **GRPO training** — reproducible recipe built on vendored `rllm` + `verl`
- 📦 **Reference checkpoint** — `VideoSEAL-8B` on HuggingFace

Index components:

- OCR subtitles (SRT) &rarr; OCR captions (+ optional embeddings)
- Clip captions (VLM) &rarr; clip captions (+ optional embeddings)
- A unified semantic index merged across modalities
- (Optional) a global `full_story.txt` summary

## 🗂️ Repository Layout

| Path | Description |
| --- | --- |
| `videoseal/` | Core Python package: agents, runners, CLI, utils |
| `scripts/` | Shell entrypoints for offline build, serving, and training |
| `rllm/`, `verl/` | Vendored RL libraries for the GRPO workflow |
| `third_party/video-subtitle-extractor/` | Vendored OCR toolchain |

## 🚀 Quick Start

### 1. Environment

```bash
conda create -n videoseal python=3.12 -y
conda activate videoseal

pip install vllm==0.11.0
pip install -e ./rllm
pip install -e ./verl
```

### 2. API keys

Export endpoints in your shell or job launcher — defaults live in `scripts/`.

```bash
export MLLM_API_KEY="sk_..."
export EMBEDDING_API_KEY="sk_..."
export AGENT_LLM_API_KEY="sk_..."
export VISUAL_INSPECT_API_KEY="sk_..."
```

### 3. Offline build

```bash
cd /path/to/Videoseal

VIDEO=/path/to/video.mp4 \
BENCHMARK=LVBench \
  ./scripts/run_offline_build.sh
```

## 🏋️ GRPO Training

The video tool-agent GRPO workflow runs out of the box thanks to the vendored `rllm` + `verl`.

```bash
cd /path/to/Videoseal

TRAIN_PARQUET='["/path/to/train.parquet"]' \
VAL_PARQUET='/path/to/val.parquet' \
MODEL_PATH='Qwen/Qwen3-8B' \
  ./scripts/train/run_video_workflow_grpo.sh train
```

Launcher: `scripts/train/run_video_workflow_grpo.sh`

## 🙏 Acknowledgements

VideoSEAL builds on excellent open-source work, including
[`rllm`](https://github.com/agentica-project/rllm),
[`verl`](https://github.com/volcengine/verl),
and the [video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor) toolchain.
