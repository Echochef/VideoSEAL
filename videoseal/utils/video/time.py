from __future__ import annotations


def sec_to_hhmmss(sec: float) -> str:
    s = max(0, int(sec))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def srt_timestamp_to_seconds(ts: str) -> float:
    s = str(ts).strip()
    if not s:
        return 0.0
    s = s.replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
        elif len(parts) == 2:
            hh = 0
            mm = int(parts[0])
            ss = float(parts[1])
        else:
            return float(s)
        return hh * 3600 + mm * 60 + ss
    except Exception:
        import re

        s2 = re.sub(r"[^0-9:.,]", "", str(ts))
        if s2 != s:
            return srt_timestamp_to_seconds(s2)
        return 0.0

