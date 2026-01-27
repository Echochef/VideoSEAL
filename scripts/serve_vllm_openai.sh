#!/usr/bin/env bash
set -euo pipefail

# Start multi-instance OpenAI-compatible vLLM servers (one per GPU).
#
# Usage (env-driven):
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   VLLM_PORT=18080 VLLM_HOST=127.0.0.1 \
#   ./scripts/serve_vllm_openai.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON="${PYTHON:-/root/miniconda3/envs/rllm/bin/python}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$(basename "${VLLM_MODEL%/}")}"
export VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-18080}"
export MAX_VLLM_SERVERS="${MAX_VLLM_SERVERS:-4}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR"
cd "$REPO_DIR"
exec "${PYTHON}" -m videoseal.cli.serve_vllm_openai_multi "$@"
