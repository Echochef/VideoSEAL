from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

from videoseal.utils.agent.env import env_flag
from videoseal.utils.video.io import get_video_duration


def cleanup_dir_tree(path: Path, *, enabled: bool, label: str) -> None:
    if not enabled:
        return
    if label == "FRAMES" and "CLEAN_FRAMES" in os.environ and not env_flag("CLEAN_FRAMES"):
        return
    try:
        if not path.exists():
            return
        if label == "FRAMES" and env_flag("CLEAN_FRAMES_ASYNC"):
            _enqueue_cleanup_dir(path, label=label)
            return
        shutil.rmtree(path)
        print(f"[CLEAN][{label}] Removed dir: {path}", file=sys.stderr)
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"[WARN] Failed to clean {label} dir {path}: {ex}", file=sys.stderr)


_CLEANUP_QUEUE: queue.SimpleQueue[tuple[Path, str, str]] | None = None
_CLEANUP_THREAD: threading.Thread | None = None


def _cleanup_worker() -> None:
    assert _CLEANUP_QUEUE is not None
    while True:
        delete_path, label, display_path = _CLEANUP_QUEUE.get()
        _cleanup_dir_tree_low_priority(delete_path, label=label, display_path=display_path)


def _ensure_cleanup_worker_started() -> queue.SimpleQueue[tuple[Path, str, str]]:
    global _CLEANUP_QUEUE, _CLEANUP_THREAD
    if _CLEANUP_QUEUE is None:
        _CLEANUP_QUEUE = queue.SimpleQueue()
        _CLEANUP_THREAD = threading.Thread(target=_cleanup_worker, name="separator_cleanup", daemon=True)
        _CLEANUP_THREAD.start()
    return _CLEANUP_QUEUE


def _cleanup_dir_tree_low_priority(path: Path, *, label: str, display_path: str) -> None:
    try:
        if not path.exists():
            return
        _rm_rf_low_priority(path, label=label)
        print(f"[CLEAN][{label}] Removed dir: {display_path}", file=sys.stderr)
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"[WARN] Failed to clean {label} dir {display_path}: {ex}", file=sys.stderr)


def _enqueue_cleanup_dir(path: Path, *, label: str) -> None:
    q = _ensure_cleanup_worker_started()
    display_path = str(path)
    delete_path = path
    try:
        if path.exists():
            delete_path = path.with_name(f".del_{path.name}_{uuid4().hex[:8]}")
            path.rename(delete_path)
    except FileNotFoundError:
        return
    except Exception:
        delete_path = path
    q.put((delete_path, label, display_path))


def _rm_rf_low_priority(path: Path, *, label: str) -> None:
    if shutil.which("rm") is None:
        shutil.rmtree(path)
        return
    cmd: list[str] = []
    if shutil.which("ionice"):
        cmd += ["ionice", "-c3"]
    if shutil.which("nice"):
        cmd += ["nice", "-n", "19"]
    cmd += ["rm", "-rf", "--", str(path)]
    t0 = time.time()
    res = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        tail = (res.stderr or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        raise RuntimeError(f"rm -rf exited with code={res.returncode}; stderr_tail={tail}")
    if (time.time() - t0) > 30.0:
        print(f"[WARN] Slow cleanup ({label}) rm -rf took {time.time()-t0:.2f}s: {path}", file=sys.stderr)


def normalize_window_min_width(video_path: str, start_sec: float, end_sec: float, *, min_width: float = 15.0) -> tuple[float, float]:
    try:
        duration = float(get_video_duration(video_path)) if video_path else 0.0
    except Exception:
        duration = 0.0

    s, e = float(start_sec), float(end_sec)
    if e <= s:
        e = s + 1e-3

    width_cap = float(min_width) if duration <= 0.0 else float(min(float(min_width), duration))
    if (e - s) < width_cap:
        center = 0.5 * (s + e)
        s = center - 0.5 * width_cap
        e = center + 0.5 * width_cap

    if duration > 0.0:
        if s < 0.0:
            s = 0.0
            e = min(duration, s + max(width_cap, 1e-3))
        if e > duration:
            e = duration
            s = max(0.0, e - max(width_cap, 1e-3))

    if e <= s:
        e = (s + 1e-3) if (duration <= 0.0) else min(duration, s + 1e-3)
    return float(max(0.0, s)), float(max(s + 1e-3, e))


def balanced_select_frames(
    *,
    per_span_frames: List[List[str]],
    per_span_ts: List[List[float]],
    span_windows: List[tuple[float, float]],
    max_total: int,
    fps_fallback: float,
    global_order: bool,
) -> tuple[List[str], List[float]]:
    selected_per_span: List[List[str]] = [[] for _ in per_span_frames]
    selected_ts_per_span: List[List[float]] = [[] for _ in per_span_frames]

    total_frames = sum(len(x) for x in per_span_frames)
    if total_frames <= max_total:
        for i in range(len(per_span_frames)):
            selected_per_span[i] = list(per_span_frames[i])
            selected_ts_per_span[i] = list(per_span_ts[i])
    else:
        idxs = [0 for _ in per_span_frames]
        taken = 0
        n = len(per_span_frames)
        while taken < max_total and any(idxs[j] < len(per_span_frames[j]) for j in range(n)):
            for j in range(n):
                if taken >= max_total:
                    break
                k = idxs[j]
                if k >= len(per_span_frames[j]):
                    continue
                selected_per_span[j].append(per_span_frames[j][k])
                tsv = per_span_ts[j][k] if (k < len(per_span_ts[j])) else None
                if tsv is None:
                    try:
                        tsv = float(span_windows[j][0]) + float(k) / max(float(fps_fallback), 1e-6)
                    except Exception:
                        tsv = float(span_windows[j][0])
                selected_ts_per_span[j].append(float(tsv))
                idxs[j] += 1
                taken += 1

    frames_all: List[str] = []
    ts_all: List[float] = []
    if global_order:
        items: List[tuple[float, str]] = []
        for i, lst in enumerate(selected_per_span):
            for kk, fp in enumerate(lst):
                ts = selected_ts_per_span[i][kk] if kk < len(selected_ts_per_span[i]) else float(span_windows[i][0])
                items.append((float(ts), fp))
        items.sort(key=lambda x: x[0])
        for ts, fp in items:
            frames_all.append(fp)
            ts_all.append(float(ts))
    else:
        for i, lst in enumerate(selected_per_span):
            frames_all.extend(lst)
            ts_all.extend(selected_ts_per_span[i])
    return frames_all, ts_all


def slugify(stem: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (stem or "").strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()


def video_id_from_path(video_path: str) -> str:
    return slugify(Path(video_path).stem)


def parse_hhmmss_spans_from_text(text: str) -> List[Tuple[str, str]]:
    s = (text or "").strip()
    if not s:
        return []
    pat = re.compile(r"(\\d{2}:\\d{2}:\\d{2})\\s*[\\-–]\\s*(\\d{2}:\\d{2}:\\d{2})")
    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for m in pat.finditer(s):
        a, b = m.group(1), m.group(2)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def select_time_diverse_windows(windows: List[Tuple[float, float]], k: int, min_gap_sec: float) -> List[int]:
    n = len(windows)
    if k <= 0 or n == 0:
        return []
    limit = min(n, k)
    if min_gap_sec <= 0:
        return list(range(limit))

    selected: List[int] = []
    centers: List[float] = []
    for idx, (s, e) in enumerate(windows):
        c = 0.5 * (float(s) + float(e))
        if any(abs(c - oc) < min_gap_sec for oc in centers):
            continue
        selected.append(idx)
        centers.append(c)
        if len(selected) >= limit:
            break
    if not selected:
        selected = list(range(limit))
    if len(selected) < limit:
        used = set(selected)
        for i in range(n):
            if i in used:
                continue
            selected.append(i)
            used.add(i)
            if len(selected) >= limit:
                break
    return selected[:limit]


def make_unique_run_dir(base: Path, *, video_id: str, prefix: str) -> Path:
    tag = f"{(prefix or 'run')}_{os.getpid()}_{uuid4().hex[:8]}"
    return Path(base) / str(video_id) / tag


def caps_to_texts(captions: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k, v in (captions or {}).items():
        if k == "subject_registry":
            continue
        text = str((v or {}).get("caption") or "").strip() if isinstance(v, dict) else ""
        if text:
            out.append((k, text))
    return out

