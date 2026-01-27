from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from rllm.rewards.reward_types import RewardOutput


FINAL_RE = re.compile(r"<final>([\s\S]*?)</final>", flags=re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", flags=re.IGNORECASE)
CHOICE_SPLIT_RE = re.compile(r"[\s,，、&|/]+")
TOOL_RESPONSE_RE = re.compile(r"<tool_response>([\s\S]*?)</tool_response>", flags=re.IGNORECASE)
SEARCH_MORE_RE = re.compile(r"\bsearch_more\b", flags=re.IGNORECASE)
MCQ_LABEL_RE = r"[A-Z0-9]{1,8}"
MCQ_LABEL_LIST_RE = rf"{MCQ_LABEL_RE}(?:\s*[,，、&|/]\s*{MCQ_LABEL_RE})*"
MCQ_ONLY_RE = re.compile(rf"(?is)^\s*[\(\[（【]?\s*({MCQ_LABEL_LIST_RE})\s*[\)\]）】]?\s*[.!。]?\s*$")
MCQ_ANSWER_IS_RE = re.compile(
    rf"(?is)(?:^|\b)(?:final\s+)?(?:the\s+)?answer\s*(?:is|=|:|：)\s*({MCQ_LABEL_LIST_RE})"
)
MCQ_ANSWER_ZH_RE = re.compile(rf"(?is)答案\s*(?:是|=|:|：)\s*({MCQ_LABEL_LIST_RE})")
MCQ_PAREN_RE = re.compile(rf"(?is)[\(\[（【]\s*({MCQ_LABEL_LIST_RE})\s*[\)\]）】]")

_LLM_JUDGE_PROMPT_CACHE: str | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_last_tag(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    s = matches[-1].group(1).strip()
    return s or None


def _extract_final(text: str) -> str | None:
    return _extract_last_tag(FINAL_RE, text or "")


def _extract_answer_tag(text: str) -> str | None:
    return _extract_last_tag(ANSWER_RE, text or "")


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_answer_from_json_obj(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k in ("answer_full", "answer", "final", "response"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        out = obj.get("output")
        if isinstance(out, str) and out.strip():
            return out.strip()
        return _extract_answer_from_json_obj(out)
    if isinstance(obj, list):
        for it in obj:
            v = _extract_answer_from_json_obj(it)
            if v:
                return v
    return None


def _extract_answer_fallback(tool_out_str: str) -> str | None:
    s = str(tool_out_str or "").strip()
    if not s:
        return None

    # 1) Tag-based format (preferred): <answer>...</answer>
    v = _extract_answer_tag(s)
    if v:
        return v

    # 2) Wrapped tool_response JSON (common in some ToolEnvs).
    m = TOOL_RESPONSE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            obj = None
        v = _extract_answer_from_json_obj(obj)
        if v:
            return v

    # 3) Plain JSON output (e.g., {"answer":"A"}).
    try:
        obj2 = json.loads(s)
    except json.JSONDecodeError:
        obj2 = None
    v = _extract_answer_from_json_obj(obj2)
    if v:
        return v

    # 4) Fallback to <final> (legacy) or plain text.
    v = _extract_final(s)
    if v:
        return v
    return s or None


def _normalize_choice_token(token: str) -> str:
    t = str(token or "").strip().upper()
    if not t:
        return ""
    t = re.sub(r"^[^A-Z0-9]+", "", t)
    t = re.sub(r"[^A-Z0-9]+$", "", t)
    return t

def _extract_answer_from_spans_call(extra_info: Dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """Return (args_dict, tool_output_str) for the last answer_from_spans call in transcript."""
    transcript = (extra_info or {}).get("transcript")
    if not isinstance(transcript, list):
        return None, None

    last_args: dict[str, Any] | None = None
    last_out: str | None = None

    for step in transcript:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        tool_outputs = step.get("tool_outputs")
        if not isinstance(tool_calls, list):
            continue

        # Case A: rLLM ToolEnvironment transcript: tool_calls(list[openai-style]) + tool_outputs(dict[id->str])
        if isinstance(tool_outputs, dict):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else None
                name = str((fn or {}).get("name") or call.get("name") or "")
                if name != "answer_from_spans":
                    continue
                args_raw = (fn or {}).get("arguments", call.get("arguments", {}))
                args = _parse_json_dict(args_raw)

                call_id = str(call.get("id") or "")
                out_str = None
                if call_id and call_id in tool_outputs:
                    out_str = str(tool_outputs.get(call_id) or "")
                elif len(tool_outputs) == 1:
                    out_str = str(next(iter(tool_outputs.values())) or "")

                last_args = args
                last_out = out_str

        # Case B: agent-style transcript: tool_calls(list[{name,arguments}]) + tool_outputs(dict-like payload)
        if not isinstance(tool_outputs, dict):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            if name != "answer_from_spans":
                continue
            args_raw = call.get("arguments", {})
            last_args = _parse_json_dict(args_raw)

            out_val = None
            if tool_outputs.get("ok") is True:
                out_val = tool_outputs.get("output")
            if out_val is None:
                out_val = tool_outputs.get("output") if "output" in tool_outputs else None
            if out_val is not None:
                last_out = str(out_val)

    return last_args, last_out


def _has_any_tool_error(extra_info: Dict[str, Any] | None) -> bool:
    """Strict policy: any tool failure forces reward to 0."""
    transcript = (extra_info or {}).get("transcript")
    if not isinstance(transcript, list):
        return False

    for step in transcript:
        if not isinstance(step, dict):
            continue
        tool_outputs = step.get("tool_outputs")
        if not isinstance(tool_outputs, dict):
            continue
        for _, val in tool_outputs.items():
            s = str(val or "")

            m = TOOL_RESPONSE_RE.search(s)
            if m:
                try:
                    obj = json.loads(m.group(1))
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and (obj.get("ok") is False):
                    return True

            if s.strip().lower().startswith("error:"):
                return True

    return False


def _has_malformed_tool_call(extra_info: Dict[str, Any] | None) -> bool:
    """Detect attempted tool calls without a following tool_outputs step."""
    transcript = (extra_info or {}).get("transcript")
    if not isinstance(transcript, list) or not transcript:
        return False

    n = len(transcript)
    for i, step in enumerate(transcript):
        if not isinstance(step, dict):
            continue
        ar = step.get("assistant_response")
        if not (isinstance(ar, str) and "<tool_call>" in ar):
            continue
        next_step = transcript[i + 1] if (i + 1) < n else None
        if not (isinstance(next_step, dict) and isinstance(next_step.get("tool_outputs"), dict) and next_step.get("tool_outputs")):
            return True
    return False


def _parse_hhmmss_to_seconds(value: Any) -> float | None:
    """Parse a time value into seconds.

    Supports:
      - numeric seconds
      - strings like "12.3"
      - strings like "HH:MM:SS" or "MM:SS"
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    s = str(value).strip()
    if not s:
        return None

    try:
        return float(s)
    except (TypeError, ValueError):
        pass

    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
            return float(mm) * 60.0 + float(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    except (TypeError, ValueError):
        return None
    return None


def _normalize_span(start_sec: float, end_sec: float) -> tuple[float, float] | None:
    s = float(start_sec)
    e = float(end_sec)
    if e < s:
        s, e = e, s
    if e - s <= 1e-9:
        return None
    return (s, e)


def _extract_time_spans(obj: Any) -> list[tuple[float, float]]:
    """Best-effort extraction of time spans from common LVU/LVBench formats."""
    spans: list[tuple[float, float]] = []
    if obj is None:
        return spans

    # String formats like "01:58-02:46; 03:10-03:20"
    if isinstance(obj, str):
        s = str(obj).strip()
        if not s:
            return spans
        s = s.replace("–", "-").replace("—", "-")
        parts = re.split(r"[;,|]+", s)
        for part in parts:
            part = part.strip()
            if not part or "-" not in part:
                continue
            a, b = part.split("-", 1)
            s_sec = _parse_hhmmss_to_seconds(a.strip())
            e_sec = _parse_hhmmss_to_seconds(b.strip())
            if s_sec is None or e_sec is None:
                continue
            span = _normalize_span(s_sec, e_sec)
            if span is not None:
                spans.append(span)
        return spans

    # One span as [start, end]
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and not isinstance(obj[0], (dict, list, tuple)):
        s = _parse_hhmmss_to_seconds(obj[0])
        e = _parse_hhmmss_to_seconds(obj[1])
        if s is not None and e is not None:
            span = _normalize_span(s, e)
            if span is not None:
                spans.append(span)
        return spans

    # List/tuple of items
    if isinstance(obj, (list, tuple)):
        for it in obj:
            spans.extend(_extract_time_spans(it))
        return spans

    # Dict formats
    if isinstance(obj, dict):
        # {start, end} or {start_time, end_time}
        if any(k in obj for k in ("start", "start_time", "begin")) and any(k in obj for k in ("end", "end_time", "finish", "stop")):
            s = obj.get("start")
            if s is None:
                s = obj.get("start_time", obj.get("begin"))
            e = obj.get("end")
            if e is None:
                e = obj.get("end_time", obj.get("finish", obj.get("stop")))
            s_sec = _parse_hhmmss_to_seconds(s)
            e_sec = _parse_hhmmss_to_seconds(e)
            if s_sec is not None and e_sec is not None:
                span = _normalize_span(s_sec, e_sec)
                if span is not None:
                    spans.append(span)
            return spans

        # Point timestamp – treat as a tiny span
        if "timestamp_sec" in obj or "timestamp" in obj or "time" in obj:
            v = obj.get("timestamp_sec", obj.get("timestamp", obj.get("time")))
            t = _parse_hhmmss_to_seconds(v)
            if t is not None:
                span = _normalize_span(t, t + 1e-3)
                if span is not None:
                    spans.append(span)
            return spans

        # Nested dict – recurse
        for _, v in obj.items():
            spans.extend(_extract_time_spans(v))
        return spans

    return spans


def _get_time_reference_spans(extra_info: Dict[str, Any] | None) -> list[tuple[float, float]]:
    """Extract reference time spans from task_info / extra_info.

    Expected locations:
      - meta_json.clue_intervals (preferred when available)
      - time_reference (e.g., LVBench)
    """
    if not isinstance(extra_info, dict):
        return []

    meta_raw = extra_info.get("meta_json")
    if meta_raw is None:
        inner_extra = extra_info.get("extra_info")
        if isinstance(inner_extra, dict):
            meta_raw = inner_extra.get("meta_json")

    if meta_raw is not None:
        try:
            meta_obj = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        except json.JSONDecodeError:
            meta_obj = None
        if isinstance(meta_obj, dict) and "clue_intervals" in meta_obj:
            spans = _extract_time_spans(meta_obj.get("clue_intervals"))
            if spans:
                return spans

    candidate = extra_info.get("time_reference")
    if candidate is None:
        inner_extra = extra_info.get("extra_info")
        if isinstance(inner_extra, dict):
            candidate = inner_extra.get("time_reference")
    return _extract_time_spans(candidate)


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    items = sorted([(float(s), float(e)) for (s, e) in spans], key=lambda x: (x[0], x[1]))
    out: list[tuple[float, float]] = []
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        out.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def _intervals_total_len(spans: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, float(e) - float(s)) for (s, e) in spans))


def _intervals_intersection_len(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return 0.0
    a_m = _merge_intervals(a)
    b_m = _merge_intervals(b)
    i = 0
    j = 0
    inter = 0.0
    while i < len(a_m) and j < len(b_m):
        a_s, a_e = a_m[i]
        b_s, b_e = b_m[j]
        left = max(a_s, b_s)
        right = min(a_e, b_e)
        if right > left:
            inter += right - left
        if a_e <= b_e:
            i += 1
        else:
            j += 1
    return float(inter)


def _answer_from_spans_span_gate(extra_info: Dict[str, Any] | None) -> dict[str, Any] | None:
    """Return span-gate metadata when answer_from_spans is used; None when not present.

    Gate rule:
      - primary_span duration must be <= 16s
      - sum(other_spans durations) must be <= 512s
    """
    args, _ = _extract_answer_from_spans_call(extra_info)
    if args is None:
        return None

    primary_raw = args.get("primary_span")
    other_raw = args.get("other_spans", [])

    primary_spans = _extract_time_spans(primary_raw)
    other_spans = _extract_time_spans(other_raw)

    primary_dur = None
    if primary_spans:
        (ps, pe) = primary_spans[0]
        primary_dur = float(pe - ps)
    other_total = float(sum((e - s) for (s, e) in other_spans))

    ok = True
    reason = ""
    if primary_dur is None:
        ok = False
        reason = "missing_primary_span"
    elif primary_dur > 16.0:
        ok = False
        reason = "primary_span_too_long"
    elif other_total > 512.0:
        ok = False
        reason = "other_spans_too_long"

    return {
        "answer_from_spans": True,
        "answer_from_spans_primary_dur_sec": primary_dur,
        "answer_from_spans_other_total_sec": other_total,
        "answer_from_spans_span_ok": bool(ok),
        "answer_from_spans_span_reason": reason,
        "answer_from_spans_primary_max_sec": 16.0,
        "answer_from_spans_other_max_total_sec": 512.0,
    }


def _is_mcq_ground(ground_truth: Any) -> bool:
    s = str(ground_truth or "").strip().upper()
    if not s:
        return False
    tokens = [_normalize_choice_token(t) for t in CHOICE_SPLIT_RE.split(s)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return False
    return all(re.fullmatch(r"[A-Z0-9]+", t) for t in tokens)


def _extract_choices(text: str) -> set[str]:
    s = str(text or "").strip().upper()
    if not s:
        return set()
    out: set[str] = set()
    for tok in CHOICE_SPLIT_RE.split(s):
        t = _normalize_choice_token(tok)
        if not t:
            continue
        if re.fullmatch(r"[A-Z0-9]+", t):
            out.add(t)
    return out


def _mcq_exact_match(pred: str, ground_truth: Any) -> bool:
    gt_set = _extract_choices(ground_truth)
    pred_set = _extract_choices(pred)
    return bool(gt_set) and pred_set == gt_set


def _extract_mcq_answer_text(text: str) -> tuple[str, str]:
    s = str(text or "").strip()
    if not s:
        return "", "empty"

    # (A) / A / A. / (A/B)
    m = MCQ_ONLY_RE.match(s)
    if m:
        return str(m.group(1)).strip(), "only"

    # The answer is A / Answer: A/B
    m = MCQ_ANSWER_IS_RE.search(s)
    if m:
        return str(m.group(1)).strip(), "answer_is"

    # 答案是A / 答案：A/B
    m = MCQ_ANSWER_ZH_RE.search(s)
    if m:
        return str(m.group(1)).strip(), "answer_zh"

    # Parentheses anywhere; prefer the last one.
    matches = list(MCQ_PAREN_RE.finditer(s))
    if matches:
        return str(matches[-1].group(1)).strip(), "paren"

    # Last resort: pick the last short alnum token.
    tokens = re.findall(r"[A-Za-z0-9]{1,8}", s)
    tokens = [t.upper() for t in tokens if t]
    if not tokens:
        return s, "raw"

    single_letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
    if single_letters:
        return single_letters[-1], "last_single"

    short_tokens = [t for t in tokens if len(t) <= 2]
    if short_tokens:
        return short_tokens[-1], "last_short"

    return tokens[-1], "last_token"


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+", text or "")


def _rouge_l_f1(ref_tokens: list[str], pred_tokens: list[str]) -> float:
    if not ref_tokens or not pred_tokens:
        return 0.0
    n, m = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[n][m]
    recall = lcs / n if n else 0.0
    precision = lcs / m if m else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _open_ended_score(pred: str, ground_truth: Any) -> float:
    gt = str(ground_truth or "").strip()
    if not gt:
        return 0.0
    pred = str(pred or "").strip()

    ref_words = _tokenize_words(gt)
    pred_words = _tokenize_words(pred)
    if ref_words and pred_words:
        score = _rouge_l_f1(ref_words, pred_words)
        if score > 0.0 and (len(ref_words) >= 2 and len(pred_words) >= 2):
            return score
    return _rouge_l_f1(list(gt), list(pred))


def _count_tool_calls(extra_info: Dict[str, Any] | None) -> int:
    if not isinstance(extra_info, dict):
        return 0
    raw_calls = extra_info.get("tool_calls_total")
    if raw_calls is not None:
        try:
            return int(raw_calls)
        except (TypeError, ValueError):
            pass

    transcript = extra_info.get("transcript")
    if not isinstance(transcript, list):
        return 0
    total = 0
    for step in transcript:
        if isinstance(step, dict) and isinstance(step.get("tool_calls"), list):
            total += len(step.get("tool_calls") or [])
    return int(total)


def _parse_tool_output_payload(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    s = str(raw or "").strip()
    if not s:
        return None
    m = TOOL_RESPONSE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    if s.lower().startswith("error:"):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _extract_hit_timestamps(payload: Any) -> list[float]:
    hits: list[float] = []
    if isinstance(payload, dict):
        raw_hits = payload.get("hits")
    elif isinstance(payload, list):
        raw_hits = payload
    else:
        return hits
    if not isinstance(raw_hits, list):
        return hits
    for item in raw_hits:
        t = None
        if isinstance(item, (int, float)):
            t = float(item)
        elif isinstance(item, str):
            t = _parse_hhmmss_to_seconds(item)
        elif isinstance(item, dict):
            val = item.get("timestamp_sec", item.get("timestamp", item.get("time")))
            t = _parse_hhmmss_to_seconds(val)
        if t is not None:
            hits.append(float(t))
    return hits


def _iter_tool_calls(extra_info: Dict[str, Any] | None) -> list[tuple[str, dict[str, Any], Any]]:
    transcript = (extra_info or {}).get("transcript")
    if not isinstance(transcript, list):
        return []
    out: list[tuple[str, dict[str, Any], Any]] = []
    for step in transcript:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        tool_outputs = step.get("tool_outputs")
        if not isinstance(tool_calls, list) or not isinstance(tool_outputs, dict):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else None
            name = str((fn or {}).get("name") or call.get("name") or "")
            args_raw = (fn or {}).get("arguments", call.get("arguments", {}))
            args = _parse_json_dict(args_raw)
            call_id = str(call.get("id") or "")
            out_raw = None
            if call_id and call_id in tool_outputs:
                out_raw = tool_outputs.get(call_id)
            elif len(tool_outputs) == 1:
                try:
                    out_raw = next(iter(tool_outputs.values()))
                except Exception:
                    out_raw = None
            out.append((name, args, out_raw))
    return out


def _contains_search_more(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(SEARCH_MORE_RE.search(value))
    if isinstance(value, dict):
        for k, v in value.items():
            if _contains_search_more(k) or _contains_search_more(v):
                return True
        return False
    if isinstance(value, (list, tuple)):
        for item in value:
            if _contains_search_more(item):
                return True
        return False
    return bool(SEARCH_MORE_RE.search(str(value)))


def _last_visual_inspect_search_more(extra_info: Dict[str, Any] | None) -> bool:
    calls = _iter_tool_calls(extra_info)
    if not calls:
        return False
    for name, _, out_raw in reversed(calls):
        tool_name = str(name or "").strip()
        if not tool_name:
            continue
        if tool_name == "finish":
            continue
        if tool_name not in {"visual_inspect", "visual_inspect_alias"}:
            return False
        return _contains_search_more(out_raw)
    return False


def _get_last_tool_name(extra_info: Dict[str, Any] | None) -> str | None:
    calls = _iter_tool_calls(extra_info)
    if not calls:
        return None
    for name, _, _ in reversed(calls):
        tool_name = str(name or "").strip()
        if not tool_name:
            continue
        if tool_name == "finish":
            continue
        return tool_name
    return None


def _collect_tool_time_signals(extra_info: Dict[str, Any] | None) -> dict[str, Any]:
    hit_timestamps: list[float] = []
    all_spans: list[tuple[float, float]] = []
    last_vis_spans: list[tuple[float, float]] = []

    span_eps = _env_float("VIDEO_REWARD_TOOL_HIT_SPAN_EPS", 1.0)
    if span_eps <= 0:
        span_eps = 1e-3

    for name, args, out_raw in _iter_tool_calls(extra_info):
        tool_name = str(name or "")
        payload = _parse_tool_output_payload(out_raw)

        if tool_name == "image_retrieve":
            hits = _extract_hit_timestamps(payload)
            if hits:
                hit_timestamps.extend(hits)
                for t in hits:
                    span = _normalize_span(float(t), float(t) + span_eps)
                    if span is not None:
                        all_spans.append(span)
            continue

        if tool_name in {"visual_retrieve", "semantic_retrieve"}:
            spans_payload = None
            if isinstance(payload, dict):
                if "useful_spans" in payload:
                    spans_payload = payload.get("useful_spans")
                elif "spans" in payload:
                    spans_payload = payload.get("spans")
            elif isinstance(payload, list):
                spans_payload = payload
            spans = _extract_time_spans(spans_payload)
            if spans:
                all_spans.extend(spans)
            continue

        if tool_name in {"visual_inspect", "visual_inspect_alias"}:
            spans_payload = None
            if isinstance(payload, dict) and "spans" in payload:
                spans_payload = payload.get("spans")
            if spans_payload is None:
                spans_payload = (args or {}).get("spans")
            spans = _extract_time_spans(spans_payload)
            if spans:
                last_vis_spans = spans
                all_spans.extend(spans)
            continue

        if tool_name == "answer_from_spans":
            spans = _extract_time_spans((args or {}).get("primary_span"))
            spans.extend(_extract_time_spans((args or {}).get("other_spans", [])))
            if spans:
                all_spans.extend(spans)
            continue

    return {
        "hit_timestamps": hit_timestamps,
        "all_spans": all_spans,
        "last_vis_spans": last_vis_spans,
    }




def _span_f1(pred_span: tuple[float, float], ref_m: list[tuple[float, float]], ref_total: float) -> float:
    if not ref_m or ref_total <= 0.0:
        return 0.0
    ps, pe = pred_span
    pred_len = max(0.0, float(pe) - float(ps))
    if pred_len <= 0.0:
        return 0.0
    overlap = _intervals_intersection_len([(ps, pe)], ref_m)
    if overlap <= 0.0:
        return 0.0
    precision = overlap / pred_len
    recall = overlap / ref_total if ref_total > 0 else 0.0
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _visual_gate_value(vis_iou_max: float, *, mode: str, thresh: float, has_ref: bool) -> float:
    if not has_ref:
        return 1.0
    mode = str(mode or "").strip().lower()
    if mode in {"none", "off", "disable"}:
        return 1.0
    if thresh <= 0:
        return 1.0 if vis_iou_max > 0 else 0.0
    if mode in {"hard", "strict"}:
        return 1.0 if vis_iou_max >= thresh else 0.0
    return _clip01(vis_iou_max / thresh)


def _mcq_format_ok(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    m = MCQ_ONLY_RE.match(s)
    if not m:
        return False
    choices = _extract_choices(m.group(1))
    if not choices:
        return False
    for c in choices:
        if len(c) != 1:
            return False
        if not ("A" <= c <= "Z"):
            return False
    return True


def _format_ok(text: str, *, is_mcq: bool, require_final_tag: bool, final_tag_present: bool) -> bool:
    if require_final_tag and not final_tag_present:
        return False
    if is_mcq:
        return _mcq_format_ok(text)
    return bool(str(text or "").strip())


def _extract_json_object(text: str | None) -> dict[str, Any] | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _load_llm_judge_prompt() -> str:
    global _LLM_JUDGE_PROMPT_CACHE
    if _LLM_JUDGE_PROMPT_CACHE is not None:
        return _LLM_JUDGE_PROMPT_CACHE
    path = os.getenv("LLM_JUDGE_SYSTEM_PROMPT_FILE") or ""
    if not path:
        _LLM_JUDGE_PROMPT_CACHE = ""
        return _LLM_JUDGE_PROMPT_CACHE
    prompt_path = Path(path)
    if not prompt_path.exists():
        _LLM_JUDGE_PROMPT_CACHE = ""
        return _LLM_JUDGE_PROMPT_CACHE
    _LLM_JUDGE_PROMPT_CACHE = prompt_path.read_text(encoding="utf-8")
    return _LLM_JUDGE_PROMPT_CACHE


def _build_llm_judge_user_message(task_info: Dict[str, Any] | None, *, final_answer: str) -> str:
    info = task_info or {}
    question = str(info.get("question") or info.get("prompt") or info.get("input") or "").strip()
    options = info.get("options") or info.get("choices") or info.get("option_list")
    options_txt = ""
    if isinstance(options, list):
        options_txt = "\n".join([str(o) for o in options if str(o).strip()])
    elif options is not None:
        options_txt = str(options).strip()

    transcript = info.get("transcript")
    transcript_txt = ""
    if transcript is not None:
        try:
            transcript_txt = json.dumps(transcript, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            transcript_txt = str(transcript)

    parts: list[str] = []
    if question:
        parts.append(f"Question:\n{question}")
    if options_txt:
        parts.append(f"Options:\n{options_txt}")
    parts.append(f"Final:\n<final>{final_answer}</final>")
    if transcript_txt:
        parts.append(f"Transcript:\n{transcript_txt}")
    return "\n\n".join(parts)


def _llm_judge_score(task_info: Dict[str, Any] | None, *, final_answer: str) -> tuple[float, dict[str, Any]]:
    if not _env_flag("LLM_JUDGE_REWARD", default=False):
        return 0.0, {}

    api_key = os.getenv("LLM_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        return 0.0, {"llm_judge_error": "missing_api_key"}

    model = os.getenv("LLM_JUDGE_MODEL") or ""
    if not model:
        return 0.0, {"llm_judge_error": "missing_model"}

    system_prompt = _load_llm_judge_prompt()
    if not system_prompt.strip():
        return 0.0, {"llm_judge_error": "missing_system_prompt"}

    base_url = os.getenv("LLM_JUDGE_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    temperature = _env_float("LLM_JUDGE_TEMPERATURE", 0.0)
    max_tokens = _env_int("LLM_JUDGE_MAX_TOKENS", 2048)
    user_prompt = _build_llm_judge_user_message(task_info, final_answer=final_answer)

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - optional dependency
        return 0.0, {"llm_judge_error": f"missing_openai:{type(exc).__name__}"}

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return 0.0, {"llm_judge_error": f"llm_call_failed:{type(exc).__name__}"}

    content = ""
    try:
        if resp and resp.choices:
            content = str(resp.choices[0].message.content or "")
    except Exception:
        content = ""

    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        meta = {"llm_judge_error": "invalid_judge_output"}
        if _env_flag("LLM_JUDGE_DEBUG", default=False):
            meta["llm_judge_raw"] = content
        return 0.0, meta

    raw_score = parsed.get("process_score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = _clip01(score)
    meta: dict[str, Any] = {"llm_judge_score": score}
    if _env_flag("LLM_JUDGE_DEBUG", default=False):
        meta["llm_judge_raw"] = content
        meta["llm_judge_parsed"] = parsed
    return score, meta


def _compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    _ = (data_source, kwargs)

    tool_error_any = _has_any_tool_error(extra_info) or _has_malformed_tool_call(extra_info)

    ans_args, ans_out = _extract_answer_from_spans_call(extra_info)
    answer_from_spans_called = ans_args is not None
    final_tag_present = False
    answer_text_raw = ""
    if answer_from_spans_called:
        tool_out_str = str(ans_out or "")
        answer_text_raw = _extract_answer_fallback(tool_out_str) or ""
    else:
        final_txt = _extract_final(solution_str or "")
        if final_txt is not None:
            final_tag_present = True
            answer_text_raw = final_txt
        else:
            answer_text_raw = _extract_answer_tag(solution_str or "") or ""
    answer_text_raw = str(answer_text_raw or "").strip()

    is_mcq = _is_mcq_ground(ground_truth)
    correct = False
    rouge_l = 0.0
    answer_text = answer_text_raw
    answer_extraction_mode = "raw"
    if is_mcq:
        answer_text, answer_extraction_mode = _extract_mcq_answer_text(answer_text_raw)
        answer_text = str(answer_text or "").strip()
        if answer_text:
            correct = _mcq_exact_match(answer_text, ground_truth)
    else:
        rouge_l = _open_ended_score(answer_text_raw, ground_truth)
        open_match_thresh = _env_float("VIDEO_OPEN_MATCH_THRESH", 0.7)
        correct = float(rouge_l) >= open_match_thresh

    format_ok = _format_ok(
        answer_text_raw,
        is_mcq=bool(is_mcq),
        require_final_tag=not answer_from_spans_called,
        final_tag_present=final_tag_present,
    )

    answer_score = float(1.0 if (is_mcq and correct) else _clip01(rouge_l))
    if is_mcq and not correct:
        answer_score = 0.0

    metadata: Dict[str, Any] = {
        "is_mcq": bool(is_mcq),
        "answer_text": answer_text,
        "answer_text_raw": answer_text_raw,
        "answer_extraction_mode": str(answer_extraction_mode),
        "answer_score": float(answer_score),
        "answer_from_spans_called": bool(answer_from_spans_called),
        "format_ok": bool(format_ok),
    }
    if not is_mcq:
        metadata["rougeL"] = float(rouge_l)

    last_tool_name = _get_last_tool_name(extra_info)
    last_tool_is_visual_inspect = bool(last_tool_name in {"visual_inspect", "visual_inspect_alias"})
    last_visual_inspect_search_more = bool(_last_visual_inspect_search_more(extra_info))
    metadata["last_tool_name"] = str(last_tool_name or "")
    metadata["last_tool_is_visual_inspect"] = bool(last_tool_is_visual_inspect)
    metadata["last_visual_inspect_search_more"] = bool(last_visual_inspect_search_more)

    span_gate = _answer_from_spans_span_gate(extra_info)
    if isinstance(span_gate, dict):
        metadata.update(span_gate)
        if not bool(span_gate.get("answer_from_spans_span_ok", True)):
            out = {
                "score": 0.0,
                "acc": 1.0 if correct else 0.0,
                "reason": "answer_from_spans_span_violation",
            }
            out.update(metadata)
            if tool_error_any:
                out["tool_error_any"] = True
            return out

    ref_spans = _get_time_reference_spans(extra_info)
    ref_m = _merge_intervals(ref_spans)
    ref_total = _intervals_total_len(ref_m)

    tool_signals = _collect_tool_time_signals(extra_info)
    all_spans = tool_signals.get("all_spans") or []
    last_vis_spans = tool_signals.get("last_vis_spans") or []

    tool_time_hit_any = False
    tool_time_f1_best = 0.0
    if ref_m and all_spans:
        for span in all_spans:
            if _intervals_intersection_len([span], ref_m) > 0.0:
                tool_time_hit_any = True
            f1 = _span_f1(span, ref_m, ref_total)
            if f1 > tool_time_f1_best:
                tool_time_f1_best = f1

    vis_iou_max = 0.0
    if ref_m and last_vis_spans:
        for vs, ve in last_vis_spans:
            for rs, re_ in ref_m:
                iou = _interval_iou(vs, ve, rs, re_)
                if iou > vis_iou_max:
                    vis_iou_max = iou

    gate_mode = os.getenv("VIDEO_REWARD_VISUAL_GATE_MODE") or "soft"
    gate_thresh = 0.05
    gate_val = _visual_gate_value(vis_iou_max, mode=gate_mode, thresh=gate_thresh, has_ref=bool(ref_m))

    w_final = max(0.0, _env_float("VIDEO_REWARD_FINAL_WEIGHT", 0.3))
    w_tool = max(0.0, _env_float("VIDEO_REWARD_TOOL_HIT_WEIGHT", 0.2))
    w_vis = max(0.0, _env_float("VIDEO_REWARD_VISUAL_INSPECT_WEIGHT", 0.2))
    w_judge = max(0.0, _env_float("VIDEO_REWARD_JUDGE_WEIGHT", 0.1))
    w_fmt = max(0.0, _env_float("VIDEO_REWARD_FORMAT_WEIGHT", 0.1))

    reward_final = float(w_final * (1.0 if correct else 0.0) * gate_val)
    reward_tool_hit = float(w_tool * _clip01(tool_time_f1_best))

    vis_target = _env_float("VIDEO_REWARD_VISUAL_IOU_TARGET", 0.5)
    if vis_target > 0:
        reward_vis = float(w_vis * _clip01(vis_iou_max / vis_target))
    else:
        reward_vis = float(w_vis * _clip01(vis_iou_max))

    judge_s, judge_meta = _llm_judge_score(extra_info, final_answer=answer_text_raw or answer_text)
    reward_judge = float(w_judge * _clip01(judge_s))
    reward_format = float(w_fmt * (1.0 if format_ok else 0.0) * gate_val)

    tool_calls_total = _count_tool_calls(extra_info)

    core_score = reward_final + reward_tool_hit + reward_vis + reward_judge + reward_format
    score = _clip01(core_score)

    if not answer_text_raw:
        reason = "missing_final_answer"
    elif correct:
        reason = "mcq_exact_match" if is_mcq else "rougeL_above_thresh"
    else:
        reason = "incorrect_answer"

    result: Dict[str, Any] = {
        "score": float(score),
        "acc": 1.0 if correct else 0.0,
        "reason": reason,
        "time_reference_spans": ref_spans,
        "tool_time_hit_any": bool(tool_time_hit_any),
        "visual_inspect_iou_max": float(vis_iou_max),
        "reward_final_answer": float(reward_final),
        "reward_tool_time_hit": float(reward_tool_hit),
        "reward_visual_inspect": float(reward_vis),
        "reward_judge": float(reward_judge),
        "reward_format": float(reward_format),
        "tool_time_f1_best": float(tool_time_f1_best),
        "tool_calls_total": int(tool_calls_total),
        "visual_gate": float(gate_val),
    }
    result.update(metadata)
    if judge_meta:
        result.update(judge_meta)

    if not answer_text_raw:
        result["score"] = 0.0
    if not answer_from_spans_called and not final_tag_present:
        result["reason_prev"] = result.get("reason", "")
        result["reason"] = "missing_final_tag"
        result["score"] = 0.0

    if tool_error_any:
        result["reason_prev"] = result.get("reason", "")
        result["reason"] = "tool_error"
        result["tool_error_any"] = True
        result["score"] = 0.0
    elif last_visual_inspect_search_more and _env_flag("VIDEO_REWARD_ZERO_ON_SEARCH_MORE", default=False):
        result["reason_prev"] = result.get("reason", "")
        result["reason"] = "visual_inspect_search_more"
        result["score"] = 0.0
    elif (
        float(result.get("score", 0.0)) > 0.0
        and not last_tool_is_visual_inspect
        and _env_flag("VIDEO_REWARD_ZERO_ON_LAST_NOT_VISUAL_INSPECT", default=False)
    ):
        result["reason_prev"] = result.get("reason", "")
        result["reason"] = "last_tool_not_visual_inspect"
        result["score"] = 0.0
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Video reward (process): FINAL + TOOL_HIT + VISUAL_INSPECT + JUDGE + FORMAT."""
    return _compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def reward_video(task_info: dict, action: str) -> RewardOutput:
    task_info = task_info or {}
    ground_truth = task_info.get("ground_truth")
    extra_info: Dict[str, Any] = dict(task_info)
    data_source = task_info.get("data_source", "video")

    score = compute_score(
        data_source=data_source,
        solution_str=action,
        ground_truth=ground_truth,
        extra_info=extra_info if extra_info else None,
    )

    reward_val = float(score.get("score", 0.0))
    metadata = score
    try:
        acc_val = float(score.get("acc", 0.0))
        is_correct = bool(acc_val >= 0.5)
    except (TypeError, ValueError):
        is_correct = None

    return RewardOutput(reward=reward_val, metadata=metadata, is_correct=is_correct)
