from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ..video.subtitles import parse_srt_to_segments


def srt_to_captions_json(srt_path: str) -> Dict[str, Any]:
    """Convert an SRT file to a captions-like JSON used by downstream indexing."""
    segs = parse_srt_to_segments(srt_path)
    out: Dict[str, Any] = {}
    for key, text in segs.items():
        t = (text or "").strip()
        if not t:
            continue
        out[key] = {"caption": t, "entities": []}
    out["subject_registry"] = {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build captions-like JSON from SRT")
    ap.add_argument("--srt", required=True, help="Path to .srt file")
    ap.add_argument("--out", required=True, help="Path to output captions.json")
    args = ap.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.is_file():
        raise SystemExit(f"[ERROR] SRT not found: {srt_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = srt_to_captions_json(srt_path.as_posix())
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] captions(from SRT) saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

