from __future__ import annotations

from typing import List, Tuple


def fixed_windows(duration_sec: float, window_sec: int, stride_sec: int | None = None) -> List[Tuple[int, int]]:
    """Split a timeline [0, duration_sec) into fixed windows.

    Returns a list of (start_sec, end_sec) inclusive integers.
    """
    if duration_sec <= 0 or window_sec <= 0:
        return []
    stride = stride_sec or window_sec
    out: List[Tuple[int, int]] = []
    start = 0
    last = int(duration_sec)
    while start <= last:
        end = min(start + window_sec - 1, last)
        out.append((start, end))
        start += stride
    return out

