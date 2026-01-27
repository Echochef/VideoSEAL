from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict


def parse_tool_outputs(tool_outputs: Any) -> Dict[str, Any] | None:
    """Parse stringified JSON values in a tool_outputs dict into structured objects.

    Returns a new parsed dict if any field is parseable; otherwise returns None.
    """
    import json

    if not isinstance(tool_outputs, dict):
        return None
    parsed: Dict[str, Any] = {}
    for key, value in tool_outputs.items():
        if isinstance(value, str):
            s = value.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed[key] = json.loads(s)
                except Exception:
                    continue
    return parsed if parsed else None


def normalize_tool_info(raw_info: Any) -> Dict[str, Any] | None:
    """Normalize tool info for dumping into rollout JSON.

    - Preserve order: tool_calls_total, transcript, then others.
    - For transcript steps, add a parsed sibling for tool_outputs if possible.
    """
    if not isinstance(raw_info, dict) or not raw_info:
        return None

    ordered = OrderedDict()
    # fixed order keys first
    if "tool_calls_total" in raw_info:
        ordered["tool_calls_total"] = raw_info.get("tool_calls_total")

    def _normalize_transcript(transcript):
        if not isinstance(transcript, list):
            return transcript
        normalized = []
        for step in transcript:
            if not isinstance(step, dict):
                normalized.append(step)
                continue
            ordered_step = OrderedDict()
            if "tool_calls" in step:
                ordered_step["tool_calls"] = step["tool_calls"]
            tool_outputs = step.get("tool_outputs")
            if "tool_outputs" in step:
                ordered_step["tool_outputs"] = tool_outputs
                parsed = parse_tool_outputs(tool_outputs)
                existing_parsed = step.get("tool_outputs_parsed")
                parsed_value = parsed or existing_parsed
                if parsed_value is not None:
                    ordered_step["tool_outputs_parsed"] = parsed_value
            for key in step:
                if key in {"tool_calls", "tool_outputs", "tool_outputs_parsed", "tool_calls_aligned"}:
                    continue
                ordered_step[key] = step[key]
            normalized.append(ordered_step)
        return normalized

    if "transcript" in raw_info:
        ordered["transcript"] = _normalize_transcript(raw_info.get("transcript"))

    # append the rest
    for key in raw_info:
        if key in {"tool_calls_total", "transcript"}:
            continue
        ordered[key] = raw_info[key]

    return ordered if ordered else None

