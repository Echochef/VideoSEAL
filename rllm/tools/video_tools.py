from __future__ import annotations

"""Minimal LVU video tools used by Videoseal.

This repo is being open-sourced, so we keep the tool surface area small and
deterministic. Only the following tools are supported:

- `visual_retrieve` (optionally includes VLM-based summary + spans)
- `visual_inspect`

IMPORTANT: Implementations are delegated to Videoseal's canonical tools so the
behavior and prompt templates are consistent with:
`scripts/run_vllm_from_jsonl.sh` → `videoseal.cli.run_from_parquet`.
"""

from typing import Any

from rllm.tools.tool_base import Tool

from videoseal.tools.base import ToolOutput as VideosealToolOutput
from videoseal.tools.visual_tools import VisualInspectAliasTool, VisualRetrieveAliasTool

import inspect


def _filter_kwargs_for_callable(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop unexpected kwargs injected by some rLLM envs.

    Videoseal's tool implementations are strict about their forward() signatures,
    while rLLM ToolEnvironment may inject extra task context fields. We keep the
    canonical Videoseal behavior and simply ignore unknown keys here.
    """

    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in (kwargs or {}).items() if k in allowed}


class VisualRetrieveTool(Tool):
    """rLLM wrapper around Videoseal's `visual_retrieve` tool."""

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        self._impl = VisualRetrieveAliasTool(name=name, description=description)
        super().__init__(name=self._impl.name, description=self._impl.description)

    @property
    def json(self) -> dict[str, Any]:
        return self._impl.json

    def forward(self, **kwargs) -> VideosealToolOutput:
        filtered = _filter_kwargs_for_callable(self._impl.forward, kwargs)
        return self._impl.forward(**filtered)


class VisualInspectTool(Tool):
    """rLLM wrapper around Videoseal's `visual_inspect` tool."""

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        self._impl = VisualInspectAliasTool(name=name, description=description)
        super().__init__(name=self._impl.name, description=self._impl.description)

    @property
    def json(self) -> dict[str, Any]:
        return self._impl.json

    def forward(self, **kwargs) -> VideosealToolOutput:
        filtered = _filter_kwargs_for_callable(self._impl.forward, kwargs)
        return self._impl.forward(**filtered)


TOOL_MAP: dict[str, type[Tool]] = {
    "visual_retrieve": VisualRetrieveTool,
    "visual_inspect": VisualInspectTool,
}


__all__ = [
    "VisualRetrieveTool",
    "VisualInspectTool",
    "TOOL_MAP",
]
