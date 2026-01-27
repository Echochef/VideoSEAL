#!/usr/bin/env python
"""GRPO training entry for multi-turn video tool workflow using AgentWorkflowEngine."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict

import hydra
import ray
from omegaconf import OmegaConf

from rllm.agents.tool_agent import ToolAgent
from rllm.agents.video_system_prompts import TOOLAGENT_SYS_PROMPT
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.video_reward_dp import reward_video
from rllm.tools.video_tools import TOOL_MAP
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from verl.utils.device import is_cuda_available


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _suppress_antlr_stderr() -> None:
    """Silence the specific ANTLR version mismatch warning on stderr."""

    pattern = re.compile(r"ANTLR runtime and generated code versions disagree")

    class _StderrFilter:
        def __init__(self, inner) -> None:
            self._inner = inner

        def write(self, s: Any) -> int:
            text = s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, bytearray)) else str(s)
            if pattern.search(text):
                return 0
            return self._inner.write(s)

        def flush(self) -> None:
            self._inner.flush()

        def fileno(self) -> int:  # pragma: no cover - pass-through
            return getattr(self._inner, "fileno", lambda: getattr(sys.__stderr__, "fileno", lambda: 2)())()

        def isatty(self) -> bool:  # pragma: no cover - pass-through
            return bool(getattr(self._inner, "isatty", lambda: False)())

    sys.stderr = _StderrFilter(sys.stderr)  # type: ignore[assignment]


def _ensure_ray(config) -> None:
    if ray.is_initialized():
        return

    if config is not None and hasattr(config, "ray_init"):
        ray_init_settings = {k: v for k, v in config.ray_init.items() if v is not None}
    else:
        ray_init_settings = {}

    runtime_env = get_ppo_ray_runtime_env()
    try:
        ray.init(runtime_env=runtime_env, **ray_init_settings)
        return
    except Exception as exc:
        print(f"ray.init() failed with configured settings, falling back to local: {exc}", file=sys.stderr)

    cleaned_settings = {k: v for k, v in ray_init_settings.items() if k != "address"}
    ray.init(address="local", runtime_env=runtime_env, **cleaned_settings)


def _build_workflow_args(config) -> Dict[str, Any]:
    max_steps = int(config.rllm.agent.get("max_steps", 10))
    trajectory_timeout = config.rllm.agent.get("trajectory_timeout")

    agent_args: Dict[str, Any] = {
        "system_prompt": TOOLAGENT_SYS_PROMPT,
        "parser_name": "qwen",
        "tool_map": TOOL_MAP,
        "include_tools_prompt": _env_flag("INCLUDE_TOOLS_PROMPT", default=False),
    }

    env_args: Dict[str, Any] = {
        "reward_fn": reward_video,
        "tool_map": TOOL_MAP,
        "max_steps": max_steps,
    }

    # Optional: low-priority defaults from the shell env, used only if a task omits them.
    task_defaults: Dict[str, Any] = {}
    for k in ("VIDEO_PATH", "video_path", "VIDEO_ID", "video_id", "IMAGE_INDEX_DIR", "VIDEO_DURATION_SEC", "video_duration_sec"):
        v = os.environ.get(k)
        if v:
            task_defaults[k] = v
    if task_defaults:
        env_args["task_defaults"] = task_defaults

    timeout = trajectory_timeout if trajectory_timeout not in (None, "", "null") else config.rllm.workflow.workflow_args.get("timeout", 1e6)

    return {
        "agent_cls": ToolAgent,
        "agent_args": agent_args,
        "env_cls": ToolEnvironment,
        "env_args": env_args,
        "max_steps": max_steps,
        "timeout": timeout,
        "gamma": config.rllm.workflow.workflow_args.get("gamma", 0.0),
        "reward_bonus_coeff": config.rllm.workflow.workflow_args.get("reward_bonus_coeff", 0.0),
    }


@hydra.main(config_path="../rllm/trainer/config", config_name="agent_ppo_trainer", version_base=None)
def main(config) -> None:
    _suppress_antlr_stderr()

    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
    print(OmegaConf.to_yaml(config))
    OmegaConf.resolve(config)

    # Ensure workflow-based execution
    config.rllm.agent.name = "tool_agent"
    config.rllm.env.name = "tool"
    config.rllm.workflow.use_workflow = True
    config.rllm.workflow.name = "multi_turn_workflow"

    _ensure_ray(config)
    workflow_args = _build_workflow_args(config)

    if is_cuda_available and config.trainer.get("profile_steps"):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    ray.get(
        runner.run.remote(
            config,
            workflow_class=MultiTurnWorkflow,
            workflow_args=workflow_args,
        )
    )

    timeline_json_file = config.ray_init.get("timeline_json_file")
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
