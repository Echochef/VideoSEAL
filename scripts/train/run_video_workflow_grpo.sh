#!/usr/bin/env bash
set -euo pipefail

# Video GRPO training launcher for rllm (tool-agent workflow).
# - Put real API keys in your shell environment / job launcher, not in this file.
# - Self-contained: uses this repo's vendored rllm + verl code via PYTHONPATH.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-python}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR"

# ---------- Inference / engine defaults (can override via env) ---------- #
export RAY_memory_monitor_refresh_ms=0 # Disable Ray memory monitor warnings.
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-60000000000}" # vLLM iteration watchdog (seconds).
export VLLM_USE_V1="${VLLM_USE_V1:-1}" # vLLM V1 engine toggle.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" # vLLM attention backend.
export RLLM_MAX_TOOL_CALLS_PER_STEP="${RLLM_MAX_TOOL_CALLS_PER_STEP:-1}" # Hard cap: max tool calls per agent step.
export INCLUDE_TOOLS_PROMPT="${INCLUDE_TOOLS_PROMPT:-0}" # Include tool schemas in the prompt (debug/ablation).

# ---------- Shared MLLM runtime knobs ---------- #
# Videoseal tools use these for timeouts / retry behavior.
export MLLM_TIMEOUT="${MLLM_TIMEOUT:-300}"
export MLLM_RETRY_TIMES="${MLLM_RETRY_TIMES:-3}"
export MLLM_RETRY_DELAY="${MLLM_RETRY_DELAY:-30}"

# ---------- Repo / runtime defaults (override via env) ---------- #
export RLLM_TMPDIR="${RLLM_TMPDIR:-${REPO_DIR}/data/.cache}" # Temp dir for frames/indices/tmp outputs.
export RLLM_OPENCV_JPEG_QUALITY="${RLLM_OPENCV_JPEG_QUALITY:-}" # OpenCV JPEG quality override (empty = OpenCV default).

# ---------- Training data (parquet) ---------- #
# NOTE: train/val parquets are not shipped with this repo.
# Set these to your own parquet(s) before running.
export TRAIN_PARQUET="${TRAIN_PARQUET:-[]}" # JSON list string, e.g. ["./data/train.parquet"]
export VAL_PARQUET="${VAL_PARQUET:-}" # Single parquet path, e.g. ./data/val.parquet

# Model
# Policy/actor base model (HF repo id or local path).
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
# ---------- Per-sample meta (normally injected from parquet/task) ---------- #
# Keep these empty here unless debugging single-sample runs.
export VIDEO_PATH="${VIDEO_PATH:-}" # Absolute path to the input video file.
export VIDEO_ID="${VIDEO_ID:-}" # Video identifier (used for resolving index paths in some pipelines).
export VIDEO_DURATION_SEC="${VIDEO_DURATION_SEC:-}" # Optional duration hint for prompt injection (seconds).

# ---------- Shared knobs: frame extraction + VLM guard ---------- #
# Used by visual_inspect (frame sampling) and visual_retrieve summarizer (MLLM runtime).
export INSPECT_FPS="${INSPECT_FPS:-1}" # Frame sampling rate for extracted images.
export INSPECT_MAX_LONG_EDGE="${INSPECT_MAX_LONG_EDGE:-1280}" # Resize: max long edge (pixels).
export INSPECT_MAX_TOTAL_IMAGES="${INSPECT_MAX_TOTAL_IMAGES:-64}" # Global cap on extracted images per tool call.
export INSPECT_VLM_MAX_TOKENS="${INSPECT_VLM_MAX_TOKENS:-2048}" # Default max tokens (fed into visual_inspect below).
export INSPECT_VLM_TEMPERATURE="${INSPECT_VLM_TEMPERATURE:-0.1}" # Default temperature (fed into per-tool *_TEMPERATURE below).
export INSPECT_GLOBAL_ORDER="${INSPECT_GLOBAL_ORDER:-1}" # Preserve global time order when merging images across spans.

# ---------- Tool: visual_inspect ---------- #
export VISUAL_INSPECT_BACKEND="${VISUAL_INSPECT_BACKEND:-openai}"
export VISUAL_INSPECT_API_BASE="${VISUAL_INSPECT_API_BASE:-https://api.openai.com/v1}"
export VISUAL_INSPECT_API_KEY="${VISUAL_INSPECT_API_KEY:-sk_your_api_key}"
export VISUAL_INSPECT_MODEL="${VISUAL_INSPECT_MODEL:-qwen2.5-vl-7b-instruct}"
export VISUAL_INSPECT_MAX_LONG_EDGE="${VISUAL_INSPECT_MAX_LONG_EDGE:-${INSPECT_MAX_LONG_EDGE}}"

# Dynamic resolution knobs (same defaults as Videoseal inference runner).
export VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE="${VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE:-1}"
export VISUAL_INSPECT_TOTAL_PIXELS="${VISUAL_INSPECT_TOTAL_PIXELS:-$((20480 * 32 * 32))}"
export VISUAL_INSPECT_MIN_PIXELS="${VISUAL_INSPECT_MIN_PIXELS:-$((16 * 28 * 28))}"
export VISUAL_INSPECT_EDGE_MULTIPLE="${VISUAL_INSPECT_EDGE_MULTIPLE:-32}"

# rLLM ToolEnvironment last-step visual_inspect fallback (default on; aligns with inference runner defaults).
export RLLM_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK="${RLLM_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK:-1}"

# ---------- Tool: visual_retrieve (LVBench-style semantic index) ---------- #
# Index root (normally injected per-sample from parquet; keep empty here unless debugging)
export VISUAL_INDEX_DIR="${VISUAL_INDEX_DIR:-}" # Unified semantic index root for visual_retrieve (per-video directory).

# Embedding backend for visual_retrieve (OpenAI-compatible embeddings endpoint).
export EMBEDDING_API_BASE="${EMBEDDING_API_BASE:-https://api.openai.com/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-sk_your_api_key}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}" # Text embedding model (must match index meta).
export EMBED_TIMEOUT="${EMBED_TIMEOUT:-60}"

export VISUAL_RETRIEVE_TOPK="${VISUAL_RETRIEVE_TOPK:-30}" # Number of spans to retrieve.
export SEMANTIC_RETRIEVE_MIX="${SEMANTIC_RETRIEVE_MIX:-embed}" # Retrieval mix mode: embed | bm25 | hybrid.
export RETRIEVE_EMBED_WEIGHT="${RETRIEVE_EMBED_WEIGHT:-0.8}" # Hybrid weight: embedding.
export RETRIEVE_BM25_WEIGHT="${RETRIEVE_BM25_WEIGHT:-0.2}" # Hybrid weight: BM25.
export RETRIEVE_MIN_TIME_GAP_SEC="${RETRIEVE_MIN_TIME_GAP_SEC:-15}" # Min time gap between returned spans.

export RETRIEVE_SUMMARY_ENABLED="${RETRIEVE_SUMMARY_ENABLED:-1}"
export RETRIEVE_SUMMARY_MAX_SPANS="${RETRIEVE_SUMMARY_MAX_SPANS:-100}"
export VISUAL_RETRIEVE_RETURN_SPANS="${VISUAL_RETRIEVE_RETURN_SPANS:-0}" # If 1: include useful_spans in the tool output (otherwise summary text only).

# visual_retrieve summarizer uses VISUAL_RETRIEVE_SUM_* (text-only)
export VISUAL_RETRIEVE_SUM_BACKEND="${VISUAL_RETRIEVE_SUM_BACKEND:-${VISUAL_INSPECT_BACKEND}}"
export VISUAL_RETRIEVE_SUM_API_BASE="${VISUAL_RETRIEVE_SUM_API_BASE:-${VISUAL_INSPECT_API_BASE}}"
export VISUAL_RETRIEVE_SUM_API_KEY="${VISUAL_RETRIEVE_SUM_API_KEY:-${VISUAL_INSPECT_API_KEY}}"
export VISUAL_RETRIEVE_SUM_MODEL="${VISUAL_RETRIEVE_SUM_MODEL:-${VISUAL_INSPECT_MODEL}}"
export VISUAL_RETRIEVE_SUM_TEMPERATURE="${VISUAL_RETRIEVE_SUM_TEMPERATURE:-0.01}"
export VISUAL_RETRIEVE_SUM_MAX_TOKENS="${VISUAL_RETRIEVE_SUM_MAX_TOKENS:-800}"


# ---------- Reward knobs (process-mode video reward) ---------- #
# Video reward is a weighted sum of:
# - final answer correctness
# - tool time hit (F1 against reference time spans)
# - visual_inspect alignment (IoU against reference time spans)
# - optional LLM-as-judge score
# - format compliance
# export VIDEO_REWARD_FINAL_WEIGHT="${VIDEO_REWARD_FINAL_WEIGHT:-0.6}" # Weight: final correctness (ROUGE-L / MCQ exact match).
# export VIDEO_REWARD_TOOL_HIT_WEIGHT="${VIDEO_REWARD_TOOL_HIT_WEIGHT:-0.2}" # Weight: tool hit F1.
# export VIDEO_REWARD_VISUAL_INSPECT_WEIGHT="${VIDEO_REWARD_VISUAL_INSPECT_WEIGHT:-0.1}" # Weight: visual_inspect IoU score.
# export VIDEO_REWARD_JUDGE_WEIGHT="${VIDEO_REWARD_JUDGE_WEIGHT:-0.0}" # Weight: LLM-as-judge score.
# export VIDEO_REWARD_FORMAT_WEIGHT="${VIDEO_REWARD_FORMAT_WEIGHT:-0.1}" # Weight: format compliance.
# export VIDEO_REWARD_ZERO_ON_SEARCH_MORE="${VIDEO_REWARD_ZERO_ON_SEARCH_MORE:-1}" # If 1: zero reward when last visual_inspect says SEARCH_MORE.
# export VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT="${VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT:-1}" # If 1: zero reward when last tool is not visual_inspect.
# export VIDEO_REWARD_VISUAL_GATE_MODE="${VIDEO_REWARD_VISUAL_GATE_MODE:-soft}" # Gate final correctness by visual IoU: soft | hard.
# Ablation: gt-reward only
export VIDEO_REWARD_FINAL_WEIGHT="${VIDEO_REWARD_FINAL_WEIGHT:-1.0}" # Weight: final correctness (ROUGE-L / MCQ exact match).
export VIDEO_REWARD_TOOL_HIT_WEIGHT="${VIDEO_REWARD_TOOL_HIT_WEIGHT:-0.0}" # Weight: tool hit F1.
export VIDEO_REWARD_VISUAL_INSPECT_WEIGHT="${VIDEO_REWARD_VISUAL_INSPECT_WEIGHT:-0.0}" # Weight: visual_inspect IoU score.
export VIDEO_REWARD_JUDGE_WEIGHT="${VIDEO_REWARD_JUDGE_WEIGHT:-0.0}" # Weight: LLM-as-judge score.
export VIDEO_REWARD_FORMAT_WEIGHT="${VIDEO_REWARD_FORMAT_WEIGHT:-0.0}" # Weight: format compliance.
export VIDEO_REWARD_ZERO_ON_SEARCH_MORE="${VIDEO_REWARD_ZERO_ON_SEARCH_MORE:-0}" # If 1: zero reward when last visual_inspect says SEARCH_MORE.
export VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT="${VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT:-0}" # If 1: zero reward when last tool is not visual_inspect.
export VIDEO_REWARD_VISUAL_GATE_MODE="${VIDEO_REWARD_VISUAL_GATE_MODE:-none}" # Gate final correctness by visual IoU: soft | hard.

export VIDEO_REWARD_VISUAL_IOU_TARGET="${VIDEO_REWARD_VISUAL_IOU_TARGET:-0.1}" # IoU normalization target (higher => harder to saturate).
export VIDEO_REWARD_TOOL_HIT_SPAN_EPS="${VIDEO_REWARD_TOOL_HIT_SPAN_EPS:-1.0}" # Convert timestamps to spans: [t, t+eps].

# LLM-as-judge (optional; defaults to off)
export LLM_JUDGE_REWARD="${LLM_JUDGE_REWARD:-0}" # If 1: enable LLM judge calls (adds cost/latency).
export LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-https://openrouter.ai/api/v1}" # Judge LLM API base.
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-sk_your_api_key}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-deepseek/deepseek-v3.2}" # Judge LLM model.
export LLM_JUDGE_JSON_ONLY="${LLM_JUDGE_JSON_ONLY:-1}" # If 1: force JSON-only outputs.
export LLM_JUDGE_TEMPERATURE="${LLM_JUDGE_TEMPERATURE:-0}" # Judge sampling temperature.
export LLM_JUDGE_MAX_TOKENS="${LLM_JUDGE_MAX_TOKENS:-2048}" # Judge max tokens.
export LLM_JUDGE_SYSTEM_PROMPT_FILE="${LLM_JUDGE_SYSTEM_PROMPT_FILE:-${REPO_DIR}/rllm/agents/llm_judge_prompt.txt}" # Judge system prompt file.

# Reward shaping (PRM four-quadrant + step-wise)
# export PRM_R_CORRECT_GROUNDED="${PRM_R_CORRECT_GROUNDED:-0.7}"
# export PRM_R_WRONG_GROUNDED="${PRM_R_WRONG_GROUNDED:-0.4}"
# export PRM_R_CORRECT_UNGROUNDED="${PRM_R_CORRECT_UNGROUNDED:-0.2}"
# export PRM_R_WRONG_UNGROUNDED="${PRM_R_WRONG_UNGROUNDED:-0.0}"
# export PRM_THINK_BONUS_MAX="${PRM_THINK_BONUS_MAX:-0.05}"
# export PRM_FORMAT_BONUS="${PRM_FORMAT_BONUS:-0.05}"
# export PRM_STEP_WEIGHT="${PRM_STEP_WEIGHT:-0.3}"
# export PRM_STEP_ALPHA="${PRM_STEP_ALPHA:-1.0}"
# export PRM_STEP_BETA="${PRM_STEP_BETA:-1.0}"
# export PRM_STEP_COST="${PRM_STEP_COST:-0.0}"
# export PRM_STEP_DUP_PENALTY="${PRM_STEP_DUP_PENALTY:-0.0}"

# Open-ended QA correctness threshold (ROUGE-L)
export VIDEO_OPEN_MATCH_THRESH="${VIDEO_OPEN_MATCH_THRESH:-0.7}"


# Training sizes / PPO hyperparameters
export NNODES="${NNODES:-1}"                       # Number of nodes.
export NGPUS="${NGPUS:-8}"                         # GPUs per node.
export TRAIN_BSZ="${TRAIN_BSZ:-32}"                # Training batch size.
export VAL_BSZ="${VAL_BSZ:-64}"                    # Validation batch size.
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}" # Candidates sampled per prompt.
export TP_SIZE="${TP_SIZE:-2}"                     # Tensor-parallel size.
export TEMP="${TEMP:-0.7}"                         # Training sampling temperature.
export VAL_TEMP="${VAL_TEMP:-0.0}"                 # Validation sampling temperature.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"         # vLLM GPU memory utilization.
export PPO_MINI_BSZ="${PPO_MINI_BSZ:-8}"           # PPO mini-batch.
export PPO_MICRO_BSZ="${PPO_MICRO_BSZ:-8}"         # PPO micro-batch.
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"    # Max prompt length (tokens).
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-32768}" # Max response length (tokens).
export PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 2048))}" # vLLM max batched tokens per GPU.
export MAX_STEPS="${MAX_STEPS:-13}"                # Max tool steps per trajectory.
export TRAJ_TIMEOUT_SEC="${TRAJ_TIMEOUT_SEC:-300}" # Trajectory timeout (seconds).
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"           # Total training epochs.
export WORKFLOW_N_PARALLEL_TASKS="${WORKFLOW_N_PARALLEL_TASKS:-64}" # Parallel rollouts.

# Output dirs / logging
export ROLLOUT_OUTDIR="${ROLLOUT_OUTDIR:-${REPO_DIR}/data/grpo/rollouts}" # Rollout/episode logs.
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_pid$$}"
RUN_DIR="${ROLLOUT_OUTDIR%/}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_video_grpo.log"
echo "[LOG] Run directory: ${RUN_DIR}" >&2
echo "[LOG] Writing stdout/stderr to ${LOG_FILE}" >&2

export GRPO_CHECKPOINT_DIR="${GRPO_CHECKPOINT_DIR:-${REPO_DIR}/data/grpo/checkpoints}" # Trainer checkpoint output root.
mkdir -p "${GRPO_CHECKPOINT_DIR}"

mode="${1:-train}"
shift || true

if [[ "${mode}" == "test-tools" ]]; then
  export RLLM_VIDEO_TOOL_TESTS=1 # Enable tool integration tests (offline).
  "${PYTHON}" -m pytest -q tests/tools/test_video_tools_integration.py "$@"
  exit 0
fi

if [[ "${mode}" == "test-reward" ]]; then
  "${PYTHON}" -m pytest -q tests/rewards/test_video_reward_tool_env_integration.py "$@"
  exit 0
fi

if [[ "${mode}" == "test-reward-live" ]]; then
  # WARNING: This mode will issue real external VLM/embedding API requests and may incur cost.
  export RLLM_VIDEO_LIVE_API_TESTS=1 # Enable tests that call real external APIs.
  "${PYTHON}" -m pytest -q tests/rewards/test_video_reward_tool_env_live_api.py "$@"
  exit 0
fi

if [[ "${mode}" == "test-reward-live-multi" ]]; then
  # WARNING: This mode will issue multiple real external VLM/embedding API requests and may incur cost.
  export RLLM_VIDEO_LIVE_API_TESTS=1 # Enable tests that call real external APIs.
  export VIDEO_REWARD_LIVE_TEST_PARQUETS="${VIDEO_REWARD_LIVE_TEST_PARQUETS:-[]}" # JSON list string of parquet paths.
  export VIDEO_REWARD_LIVE_TEST_NUM_SAMPLES="${VIDEO_REWARD_LIVE_TEST_NUM_SAMPLES:-3}" # Number of sampled videos.
  "${PYTHON}" -m pytest -q tests/rewards/test_video_reward_tool_env_live_api.py "$@"
  exit 0
fi

if [[ "${mode}" != "train" ]]; then
  echo "[ERR] Unknown mode: ${mode}. Use: train | test-tools | test-reward | test-reward-live | test-reward-live-multi" >&2
  exit 2
fi

if [[ -z "${VAL_PARQUET}" ]]; then
  echo "[ERR] Missing VAL_PARQUET. Example: VAL_PARQUET=./data/val.parquet" >&2
  exit 2
fi
if [[ -z "${TRAIN_PARQUET}" || "${TRAIN_PARQUET}" == "[]" ]]; then
  echo "[ERR] Missing TRAIN_PARQUET. Example: TRAIN_PARQUET='[\"./data/train.parquet\"]'" >&2
  exit 2
fi
# data.max_prompt_length controls PPO prompt length; actor_rollout_ref.rollout controls rollout length.
"${PYTHON}" scripts/train_video_workflow_grpo.py \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.val_batch_size="${VAL_BSZ}" \
  +data.seed=42 \
  data.return_raw_chat=True \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  actor_rollout_ref.rollout.prompt_length=32748 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.max_model_len=36864 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.warmup_style=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BSZ}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode="async" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  rllm.agent.max_steps="${MAX_STEPS}" \
  rllm.agent.trajectory_timeout="${TRAJ_TIMEOUT_SEC}" \
  rllm.env.name=tool \
  rllm.agent.name=tool_agent \
  rllm.workflow.n_parallel_tasks="${WORKFLOW_N_PARALLEL_TASKS}" \
  rllm.compact_filtering.enable=true \
  trainer.critic_warmup=0.0 \
  trainer.log_episodes=true \
  trainer.episode_log_dir="${RUN_DIR}" \
  trainer.logger=[console,tensorboard] \
  trainer.project_name=video-tools \
  trainer.experiment_name=video-grpo-toolagent \
  trainer.default_local_dir="${GRPO_CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${RUN_DIR}" \
  trainer.validation_data_dir="${RUN_DIR}/val" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=10 \
  trainer.test_freq=50 \
  trainer.val_before_train=False \
  custom_reward_function.path=rllm/rewards/video_reward_dp.py \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
