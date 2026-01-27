from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .time import srt_timestamp_to_seconds


def parse_srt_to_segments(srt_path: str) -> Dict[str, str]:
    """Parse .srt into a dict: "startSec_endSec" -> text (merged lines)."""
    try:
        with open(srt_path, "r", encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh]
    except Exception:
        return {}

    result: Dict[str, str] = {}
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip().isdigit():
            i += 1
        if i >= n or "-->" not in lines[i]:
            i += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[i].split("-->")]
        i += 1
        start_sec = int(srt_timestamp_to_seconds(start_ts))
        end_sec = int(srt_timestamp_to_seconds(end_ts))
        texts: List[str] = []
        while i < n and lines[i].strip():
            texts.append(lines[i].strip())
            i += 1
        key = f"{start_sec}_{end_sec}"
        text = " ".join(texts)
        if key in result:
            result[key] += " " + text
        else:
            result[key] = text
        i += 1
    return result


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleIndex:
    segments: Sequence[SubtitleSegment]
    starts: Sequence[float]

    def text_at(self, t: float) -> str:
        if not self.segments:
            return ""
        i = bisect.bisect_right(self.starts, float(t)) - 1
        if i < 0:
            return ""
        seg = self.segments[i]
        if float(seg.start) <= float(t) < float(seg.end):
            return str(seg.text or "").strip()
        return ""


def _parse_srt_segments(srt_path: str) -> List[SubtitleSegment]:
    try:
        raw = Path(srt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    segs: List[SubtitleSegment] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i].strip().isdigit():
            i += 1
        if i >= n or "-->" not in raw[i]:
            i += 1
            continue
        start_ts, end_ts = [t.strip() for t in raw[i].split("-->")]
        i += 1
        start_sec = float(srt_timestamp_to_seconds(start_ts))
        end_sec = float(srt_timestamp_to_seconds(end_ts))
        texts: List[str] = []
        while i < n and raw[i].strip():
            texts.append(raw[i].strip())
            i += 1
        i += 1
        text = " ".join([t for t in texts if t]).strip()
        if not text:
            continue
        if end_sec <= start_sec:
            end_sec = start_sec + 1.0
        segs.append(SubtitleSegment(start=start_sec, end=end_sec, text=text))
    segs.sort(key=lambda x: (float(x.start), float(x.end), x.text))
    return segs


def _parse_json_segments(json_path: str) -> List[SubtitleSegment]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            data = data.get("segments")
        elif isinstance(data.get("subtitles"), list):
            data = data.get("subtitles")
        elif isinstance(data.get("data"), list):
            data = data.get("data")
    if not isinstance(data, list):
        return []

    segs: List[SubtitleSegment] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        start_ts = obj.get("start") or obj.get("start_time") or obj.get("begin") or obj.get("from")
        end_ts = obj.get("end") or obj.get("end_time") or obj.get("finish") or obj.get("to")
        text = obj.get("line") or obj.get("text") or obj.get("caption") or obj.get("subtitle")
        start_sec = float(srt_timestamp_to_seconds(start_ts))
        end_sec = float(srt_timestamp_to_seconds(end_ts))
        t = str(text or "").strip()
        if not t:
            continue
        segs.append(SubtitleSegment(start=start_sec, end=end_sec, text=t))

    if not segs:
        return []

    segs.sort(key=lambda x: (float(x.start), float(x.end), x.text))

    point_like_min_dur = 0.05
    point_extend_sec = 5.0
    fixed: List[SubtitleSegment] = []
    for i, seg in enumerate(segs):
        s = float(seg.start)
        e = float(seg.end)
        if e <= s + point_like_min_dur:
            next_s: Optional[float] = None
            if i + 1 < len(segs):
                try:
                    next_s = float(segs[i + 1].start)
                except Exception:
                    next_s = None
            if next_s is not None and next_s > s:
                e = min(next_s, s + point_extend_sec)
            else:
                e = s + point_extend_sec
        if e <= s:
            e = s + point_extend_sec
        fixed.append(SubtitleSegment(start=s, end=e, text=str(seg.text or "").strip()))

    return fixed


def build_subtitle_index(subtitle_path: str) -> Optional[SubtitleIndex]:
    p = str(subtitle_path or "").strip()
    if not p:
        return None
    if not os.path.isfile(p):
        return None
    ext = Path(p).suffix.lower()
    if ext == ".srt":
        segs = _parse_srt_segments(p)
    elif ext == ".json":
        segs = _parse_json_segments(p)
    else:
        return None
    if not segs:
        return None
    starts = [float(s.start) for s in segs]
    return SubtitleIndex(segments=tuple(segs), starts=tuple(starts))

