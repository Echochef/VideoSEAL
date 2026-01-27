#!/usr/bin/env bash
set -euo pipefail

# Parquet batch runner (Videoseal).
#
# Environment variables are intentionally kept in this script for quick editing.
# The list below mirrors the conventions in:
#   DynamicvideoAgentRL/scripts/run_vllm_from_jsonl.sh
#
# Note: put real secrets in your shell environment / job launcher, not in this file.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Preferred Python for this machine (override via env).
export PYTHON="${PYTHON:-/root/miniconda3/envs/rllm/bin/python}"

# Data root (LVU unified layout on this machine).
export DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT:-/mnt/shanhai-ai/qiuchenhao/data/LVU/data}"

# ---------- Retrieve / Inspect knobs ---------- #
export INSPECT_MAX_LONG_EDGE="${INSPECT_MAX_LONG_EDGE:-1280}"
export INSPECT_VLM_MAX_TOKENS="${INSPECT_VLM_MAX_TOKENS:-2048}"
export INSPECT_VLM_TEMPERATURE="${INSPECT_VLM_TEMPERATURE:-0.1}"
export INSPECT_MAX_TOTAL_IMAGES="${INSPECT_MAX_TOTAL_IMAGES:-64}"
export INSPECT_FPS="${INSPECT_FPS:-1}"
export SEMANTIC_RETRIEVE_TOPK="${SEMANTIC_RETRIEVE_TOPK:-40}"

export VISUAL_RETRIEVE_TOPK="${VISUAL_RETRIEVE_TOPK:-30}"
export RETRIEVE_SUMMARY_MAX_SPANS="${RETRIEVE_SUMMARY_MAX_SPANS:-100}"
export RETRIEVE_SUMMARY_ENABLED="${RETRIEVE_SUMMARY_ENABLED:-1}"

export AGENT_SYS_PROMPT_LANG="${AGENT_SYS_PROMPT_LANG:-en}"
export BENCHMARK="${BENCHMARK:-lvbench}"
export SEMANTIC_RETRIEVE_MIX="${SEMANTIC_RETRIEVE_MIX:-embed}"
export RETRIEVE_EMBED_WEIGHT="${RETRIEVE_EMBED_WEIGHT:-0.8}"
export RETRIEVE_BM25_WEIGHT="${RETRIEVE_BM25_WEIGHT:-0.2}"

# ---------- Shared MLLM runtime knobs ---------- #
export MLLM_BACKEND="${MLLM_BACKEND:-openai}"
export MLLM_MAX_TOKENS="${MLLM_MAX_TOKENS:-4096}"
export MLLM_TIMEOUT="${MLLM_TIMEOUT:-300}"
export MLLM_RETRY_TIMES="${MLLM_RETRY_TIMES:-3}"
export MLLM_RETRY_DELAY="${MLLM_RETRY_DELAY:-30}"
export EMBED_RETRY_TIMES="${EMBED_RETRY_TIMES:-5}"
export EMBED_RETRY_DELAY="${EMBED_RETRY_DELAY:-30}"

# ---------- Agent LLM (text; OpenAI-compatible) ---------- #
export AGENT_LLM_API_BASE="${AGENT_LLM_API_BASE:-https://openrouter.ai/api/v1}"
export AGENT_LLM_API_KEY="${AGENT_LLM_API_KEY:-sk_your_api_key}"
export AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-google/gemini-3-flash-preview}"
export AGENT_LLM_MAX_TOKENS="${AGENT_LLM_MAX_TOKENS:-2048}"
export AGENT_LLM_TEMPERATURE="${AGENT_LLM_TEMPERATURE:-0.0}"
export AGENT_LLM_TIMEOUT="${AGENT_LLM_TIMEOUT:-60}"
export AGENT_MLLM_BACKEND="${AGENT_MLLM_BACKEND:-openai}"

# ---------- Tool-side VLM (visual_inspect) ---------- #
export VISUAL_INSPECT_BACKEND="${VISUAL_INSPECT_BACKEND:-openai}"
export VISUAL_INSPECT_API_BASE="${VISUAL_INSPECT_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VISUAL_INSPECT_API_KEY="${VISUAL_INSPECT_API_KEY:-sk_your_api_key}"
export VISUAL_INSPECT_MODEL="${VISUAL_INSPECT_MODEL:-qwen2.5-vl-7b-instruct}"

# Retrieval summarizer (optional; used when summary is enabled).
export VISUAL_RETRIEVE_SUM_BACKEND="${VISUAL_RETRIEVE_SUM_BACKEND:-${VISUAL_INSPECT_BACKEND}}"
export VISUAL_RETRIEVE_SUM_API_BASE="${VISUAL_RETRIEVE_SUM_API_BASE:-${VISUAL_INSPECT_API_BASE}}"
export VISUAL_RETRIEVE_SUM_API_KEY="${VISUAL_RETRIEVE_SUM_API_KEY:-${VISUAL_INSPECT_API_KEY}}"
export VISUAL_RETRIEVE_SUM_MODEL="${VISUAL_RETRIEVE_SUM_MODEL:-${VISUAL_INSPECT_MODEL}}"
export VISUAL_RETRIEVE_SUM_TEMPERATURE="${VISUAL_RETRIEVE_SUM_TEMPERATURE:-0.01}"
export VISUAL_RETRIEVE_SUM_MAX_TOKENS="${VISUAL_RETRIEVE_SUM_MAX_TOKENS:-800}"

# Dynamic visual_inspect resolution.
export VISUAL_INSPECT_MAX_LONG_EDGE="${VISUAL_INSPECT_MAX_LONG_EDGE:-1280}"
export VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE="${VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE:-1}"
export VISUAL_INSPECT_TOTAL_PIXELS="${VISUAL_INSPECT_TOTAL_PIXELS:-$((20480 * 32 * 32))}"
export VISUAL_INSPECT_MIN_PIXELS="${VISUAL_INSPECT_MIN_PIXELS:-$((16 * 28 * 28))}"
export VISUAL_INSPECT_EDGE_MULTIPLE="${VISUAL_INSPECT_EDGE_MULTIPLE:-32}"

# Prefer tmpfs for high-concurrency frame extraction/cleanup; fall back to /tmp.
if [[ -z "${FRAMES_ROOT:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    export FRAMES_ROOT=/dev/shm/dva_frames
  else
    export FRAMES_ROOT=/tmp/dva_frames
  fi
else
  export FRAMES_ROOT
fi
export CLEAN_FRAMES="${CLEAN_FRAMES:-0}"
export CLEAN_FRAMES_ASYNC="${CLEAN_FRAMES_ASYNC:-1}"

# ---------- Embedding API (OpenAI-compatible embeddings) ---------- #
export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-https://api.yuboar.com/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-sk_your_api_key}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"

# ---------- vLLM (optional; when AGENT_LLM_BACKEND=vllm or local /v1 server is used) ---------- #
export VLLM_MODEL="${VLLM_MODEL:-/mnt/shanhai-ai/qiuchenhao/data/checkpoints/huggingface/grpocheckpoints_seq-mean-token-mean-global_step_300/actor/}"
export VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.1}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-4096}"

# Hard timeout per question (used by per_question_runner when backend=api).
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-1000}"

# Runner params
export PARQUET="${PARQUET:-/mnt/shanhai-ai/qiuchenhao/data/LVU/parquet/val_lvbench.parquet}"
export VIDEO_ID="${VIDEO_ID:-}"
export MAX_STEPS="${MAX_STEPS:-16}"
export SAVE_RUNS="${SAVE_RUNS:-${REPO_DIR}/data/trajectory_decoupled_time_notimereward/runs_lvbench_maxruns16_summary_topk30_gpt4o}"
export CONCURRENCY="${CONCURRENCY:-64}"
export CONCURRENCY_PER_SHARD="${CONCURRENCY_PER_SHARD:-}"

# Last-step fallback knobs (optional).
export AGENT_FORCE_LAST_STEP_VISUAL_INSPECT="${AGENT_FORCE_LAST_STEP_VISUAL_INSPECT:-1}"
export AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK="${AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK:-1}"
export AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE="${AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE:-mcq}"
export AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK="${AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK:-1}"
# Videoseal does NOT force "C" on parse failure; use full-video visual_inspect fallback instead.
export AGENT_PARSE_FAILURE_FALLBACK_TO_C="${AGENT_PARSE_FAILURE_FALLBACK_TO_C:-0}"

# vLLM service discovery (for local OpenAI /v1 servers).
export USE_VLLM_META_FILE="${USE_VLLM_META_FILE:-1}"
export VLLM_META_REQUIRE_MODEL_MATCH="${VLLM_META_REQUIRE_MODEL_MATCH:-1}"
export VLLM_META_FILE="${VLLM_META_FILE:-${REPO_DIR}/serve/vllm_servers.env}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR"
cd "$REPO_DIR"
exec "${PYTHON}" -m videoseal.cli.run_from_parquet "$@"
