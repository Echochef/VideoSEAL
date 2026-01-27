from __future__ import annotations

from typing import List, Tuple

# Reuse common video utilities to avoid duplication
from ..video import get_video_duration, fixed_windows


def get_video_duration_sec(video_path: str) -> float:
    """Return duration seconds via common utility."""
    return float(get_video_duration(video_path))


def chunk_ranges(duration_sec: float, chunk_seconds: int) -> List[Tuple[float, float]]:
    """
    Split [0, duration_sec) into fixed-length chunks (last one may be shorter).
    Delegates to common `fixed_windows` (which returns inclusive end ints) and
    converts to half-open intervals (start, end) for downstream samplers.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    windows = fixed_windows(duration_sec, chunk_seconds)
    chunks: List[Tuple[float, float]] = []
    for s, e_inclusive in windows:
        start = float(s)
        # Convert inclusive end -> exclusive end by +1, then clamp to duration
        end = float(min(int(duration_sec), e_inclusive + 1))
        if end <= start:
            end = start + 1.0
        chunks.append((start, end))
    return chunks

