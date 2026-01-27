from __future__ import annotations

"""System prompt for the video tool agent.

To keep open-source behavior consistent across inference and training, rLLM
reuses Videoseal's canonical tool-agent prompt (the one used by
`scripts/run_vllm_from_jsonl.sh`).
"""

import os

from videoseal.agents.tool_agent import _system_prompt as _videoseal_system_prompt


def build_toolagent_sys_prompt(lang: str | None = None) -> str:
    """Return the canonical tool-agent system prompt (en/zh)."""

    sys_lang = str(lang or os.getenv("AGENT_SYS_PROMPT_LANG", "en"))
    return str(_videoseal_system_prompt(sys_lang)).strip()


TOOLAGENT_SYS_PROMPT: str = build_toolagent_sys_prompt()


__all__ = [
    "TOOLAGENT_SYS_PROMPT",
    "build_toolagent_sys_prompt",
]
