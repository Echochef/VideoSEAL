from __future__ import annotations

import inspect
from typing import Any, Dict

from videoseal.tools.base import Tool


ALLOWED_TOOL_EXTRA_KEYS = {
    "video_id",
    "index_path",
    "video_path",
    "questions",
    "visual_index",
    "original_question",
    "prompt_type",
}


def sanitize_args_against_schema(tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
    schema = tool.json or {}
    schema_props = set(schema.get("function", {}).get("parameters", {}).get("properties", {}).keys())
    return {k: v for k, v in (args or {}).items() if (k in schema_props) or (k in ALLOWED_TOOL_EXTRA_KEYS)}


def filter_args_for_forward(tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(tool.forward)
    except Exception:
        return dict(args or {})
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(args or {})
    allowed = set(params.keys())
    return {k: v for k, v in (args or {}).items() if k in allowed}

