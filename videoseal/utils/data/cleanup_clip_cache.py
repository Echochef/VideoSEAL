from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .check_cache_status import _caption_has_error, _load_text


def collect_invalid(cache_dir: Path) -> List[Path]:
    bad: List[Path] = []
    if not cache_dir.exists():
        return bad
    for json_path in cache_dir.rglob("*.json"):
        text = _load_text(json_path)
        if not text.strip():
            bad.append(json_path)
            continue
        try:
            data = json.loads(text)
        except Exception:
            data = {"clip_description": text}
        if isinstance(data, dict) and _caption_has_error(data):
            bad.append(json_path)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove invalid clip cache chunks containing HTTP errors.")
    parser.add_argument("--cache-dir", required=True, help="Cache directory (captions_ckpt root)")
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g., BENCHMARK/slug)")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    cache_dir = cache_root / "captions_ckpt" / Path(args.video_id)
    bad_files = collect_invalid(cache_dir)
    for path in bad_files:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"[WARN] failed to remove {path}: {exc}")
    print(f"[CLEANUP] removed={len(bad_files)} dir={cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

