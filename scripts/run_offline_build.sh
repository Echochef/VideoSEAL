#!/usr/bin/env bash
set -euo pipefail

# Offline build (Videoseal): OCR + Clip -> unified semantic index (optional summary).
#
# Usage (single video):
#   VIDEO=/path/to/video.mp4 BENCHMARK=LVBench ./scripts/run_offline_build.sh
#
# Required (for clip captions via OpenAI-compatible endpoint):
#   export MLLM_BACKEND=openai
#   export MLLM_API_BASE=https://api.openai.com/v1
#   export MLLM_API_KEY=sk_your_api_key
#   export MLLM_MODEL=gpt-4o-mini
#
# Recommended: put real secrets into your shell environment / job launcher.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON="${PYTHON:-/root/miniconda3/envs/rllm/bin/python}"
export DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT:-/mnt/shanhai-ai/qiuchenhao/data/LVU/data}"

export BENCHMARK="${BENCHMARK:-LVBench}"
export VIDEO="${VIDEO:-}"
export VIDEO_ID="${VIDEO_ID:-}"

export INDEXES_ROOT="${INDEXES_ROOT:-${DEFAULT_DATA_ROOT%/}/indexes}"
export SUMMARIES_ROOT="${SUMMARIES_ROOT:-${DEFAULT_DATA_ROOT%/}/summaries}"
export CACHE_DIR="${CACHE_DIR:-${DEFAULT_DATA_ROOT%/}/cache}"
export FRAMES_ROOT="${FRAMES_ROOT:-${DEFAULT_DATA_ROOT%/}/frames}"

export EMBED_MODEL="${EMBED_MODEL:-${EMBEDDING_MODEL:-text-embedding-3-large}}"
export CLIP_LEN_SEC="${CLIP_LEN_SEC:-16}"
export CLIP_SAMPLE_FPS="${CLIP_SAMPLE_FPS:-1}"
export CLIP_WORKERS="${CLIP_WORKERS:-32}"
export CLIP_MAX_FRAMES="${CLIP_MAX_FRAMES:-}"
export VIDEO_WORKERS="${VIDEO_WORKERS:-16}"

export OVERWRITE="${OVERWRITE:-0}"
export OVERWRITE_OCR="${OVERWRITE_OCR:-0}"
export OVERWRITE_CLIP="${OVERWRITE_CLIP:-0}"
export OVERWRITE_SUMMARY="${OVERWRITE_SUMMARY:-0}"
export SKIP_OCR="${SKIP_OCR:-0}"
export SKIP_CLIP="${SKIP_CLIP:-0}"
export SKIP_SUMMARY="${SKIP_SUMMARY:-1}"
export USE_EXISTING_SRT_ONLY="${USE_EXISTING_SRT_ONLY:-1}"
export OCR_FULL_IMAGE="${OCR_FULL_IMAGE:-0}"
export OCR_EXCLUDE_CHINESE="${OCR_EXCLUDE_CHINESE:-0}"
export CLEAN_FRAMES="${CLEAN_FRAMES:-0}"

# Default: skip embedding indexes to save compute; set to 0 to build vectors.
export SKIP_CLIP_EMBED="${SKIP_CLIP_EMBED:-1}"
export SKIP_OCR_EMBED="${SKIP_OCR_EMBED:-1}"

# Clip caption VLM (OpenAI-compatible by default).
export MLLM_BACKEND="${MLLM_BACKEND:-openai}"
export MLLM_API_BASE="${MLLM_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MLLM_API_KEY="${MLLM_API_KEY:-sk_your_api_key}"
export MLLM_MODEL="${MLLM_MODEL:-qwen2.5-vl-7b-instruct}"
export MLLM_MAX_TOKENS="${MLLM_MAX_TOKENS:-4096}"
export MLLM_TIMEOUT="${MLLM_TIMEOUT:-300}"
export MLLM_RETRY_TIMES="${MLLM_RETRY_TIMES:-3}"
export MLLM_RETRY_DELAY="${MLLM_RETRY_DELAY:-30}"

# Embeddings (OpenAI-compatible).
export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-https://api.yuboar.com/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-sk_your_api_key}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR"
cd "$REPO_DIR"
exec "${PYTHON}" -m videoseal.cli.offline_build_batch "$@"
