from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


def extract_between(text: str, begin: str, end: str) -> Optional[str]:
    s = (text or "").find(begin)
    if s == -1:
        return None
    s += len(begin)
    e = (text or "").find(end, s)
    if e == -1:
        return None
    return (text or "")[s:e].strip()


def parse_tool_call_qwen(text: str) -> Optional[Dict[str, Any]]:
    begin = "<tool_call>"
    end = "</tool_call>"
    if begin not in (text or ""):
        return None
    start = (text or "").find(begin) + len(begin)
    stop = (text or "").find(end, start)
    if stop == -1:
        return None

    json_content = (text or "")[start:stop].strip()
    try:
        obj = json.loads(json_content)
    except json.JSONDecodeError:
        tolerate = (os.getenv("AGENT_TOOLCALL_TOLERATE_EXTRA_BRACES") or "").strip().lower() in ("1", "true", "yes", "on")
        if not tolerate:
            return None
        try:
            dec = json.JSONDecoder()
            obj, idx = dec.raw_decode(json_content)
            tail = json_content[idx:].strip()
            if not tail or set(tail) != {"}"}:
                return None
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def parse_answer(text: str) -> Optional[str]:
    return extract_between(text, "<answer>", "</answer>")


def parse_thinking(text: str) -> Optional[str]:
    return extract_between(text, "<thinking>", "</thinking>")


def strip_multiple_choice_options(text: str) -> str:
    t = (str(text) or "").strip()
    if not t:
        return t
    markers = list(re.finditer(r"\(([A-Za-z])\)|\[([A-Za-z])\]|\b([A-Za-z])[\)\.:：、]\s*", t))
    if len(markers) >= 2:
        return t[: markers[0].start()].strip()
    m = re.search(r"(?m)^(?:\(|\[)?[A-Za-z](?:\)|\])?[\)\.:：、]\s+", t)
    if m:
        return t[: m.start()].strip()
    return t
