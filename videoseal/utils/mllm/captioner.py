from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..api.mllm import MLLMClient
from videoseal.prompts.caption_prompts import default_caption_prompt


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # remove leading ```(json)?
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        # remove trailing ```
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    """
    Try to parse a JSON object from text that may contain code fences or extra prose.
    - Strip ```json fences
    - If direct json.loads fails, find the first '{' and parse a balanced object substring.
    """
    s = _strip_code_fences(text)
    # fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # try substring between first '{' and matching '}'
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def caption_clip_visual_only(
    images_b64: List[str],
    start_hms: str,
    end_hms: str,
    client: MLLMClient,
) -> Dict[str, Any]:
    """
    Call MLLM to generate a JSON caption for a clip (visual-only, no transcript).
    Returns a dict parsed from the model output; if parsing fails, return a
    fallback containing 'clip_description' text only.
    """
    prompt = default_caption_prompt(start_hms, end_hms)
    text = client.generate_caption(images_b64, prompt, response_json=True)
    data = _extract_json_object(text)
    try:
        if data is None:
            raise ValueError("json parse failed")
        # Ensure minimal keys
        data.setdefault("clip_start_time", start_hms)
        data.setdefault("clip_end_time", end_hms)
        data.setdefault("clip_description", "")
        data.setdefault("entities", [])
        return data
    except Exception:
        return {
            "clip_start_time": start_hms,
            "clip_end_time": end_hms,
            "clip_description": text,
            "entities": [],
        }


def merge_subject_registry(all_entities: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Very light-weight merge of entities across clips into a 'subject_registry'.
    We simply bucket by the 'name' field and aggregate appearances/identity.
    """
    registry: Dict[str, Dict[str, Any]] = {}
    for entities in all_entities:
        for ent in entities or []:
            name = (ent.get("name") or "unknown").strip()
            if name not in registry:
                registry[name] = {
                    "appearance": set(),
                    "identity": set(),
                }
            for a in ent.get("appearance") or []:
                registry[name]["appearance"].add(str(a))
            for i in ent.get("identity") or []:
                registry[name]["identity"].add(str(i))
    # convert sets to lists
    return {k: {"appearance": sorted(v["appearance"]), "identity": sorted(v["identity"])} for k, v in registry.items()}
