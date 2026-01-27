# Videoseal

This is the official code for **VideoSEAL: Separating Planning from Answer Authority for Agentic Long Video Understanding**.

Offline build utilities for long video indexing:

- OCR subtitles (SRT) → OCR captions + (optional) embeddings
- Clip captions (VLM) → clip captions + (optional) embeddings
- Merge into a unified semantic index under `indexes/semantic/<video_id>/`
- (Optional) generate a global `full_story.txt` summary

## Layout

- Shell entrypoints: `scripts/`
- Python package: `videoseal/`
- Tests: `test/`
- OCR toolchain (vendored): `third_party/video-subtitle-extractor/`

## Configuration

- Defaults live in the scripts under `scripts/`.
- Put real API keys/endpoints in your shell environment / job launcher.

## Run offline build

```bash
cd /path/to/Videoseal

export MLLM_API_KEY="sk_your_api_key"
export EMBEDDING_API_KEY="sk_your_api_key"
export AGENT_LLM_API_KEY="sk_your_api_key"
export VISUAL_INSPECT_API_KEY="sk_your_api_key"
VIDEO=/path/to/video.mp4 BENCHMARK=LVBench ./scripts/run_offline_build.sh
```

## Run tests

```bash
/root/miniconda3/envs/rllm/bin/python -m unittest discover -s test -v
```

## GRPO training (video tool workflow)

This repo vendors a minimal copy of the `rllm/` + `verl/` Python packages (under the repo root)
to make the video tool-agent GRPO workflow runnable without an extra repo checkout.

### Training environment (conda)

```bash
conda create -n videosearl python=3.12 -y
conda activate videosearl

pip install vllm==0.11.0

cd rllm
pip install -e .

cd ../verl
pip install -e .
```

Launcher:

- `scripts/train/run_video_workflow_grpo.sh`

Example:

```bash
cd /path/to/Videoseal

# Export real API keys/endpoints in your environment before launching.

TRAIN_PARQUET='["/path/to/train.parquet"]' \
VAL_PARQUET='/path/to/val.parquet' \
MODEL_PATH='Qwen/Qwen3-8B' \
./scripts/train/run_video_workflow_grpo.sh train
```

Quick checks:

```bash
./scripts/train/run_video_workflow_grpo.sh test-reward
pytest -q tests/rewards/test_video_reward_tool_env_integration.py
```
