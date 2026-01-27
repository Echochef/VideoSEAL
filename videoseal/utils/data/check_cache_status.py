from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


BAD_TOKENS = (
    "HTTPError",
    "Client Error",
    "Forbidden",
    "Too Many Requests",
    "[ERROR]",
    "HTTP ",
)


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def check_summary(summary_root: Path, video_id: Path) -> Dict[str, int]:
    summary_file = summary_root / video_id / "full_story.txt"
    if not summary_file.exists():
        return {"summary_missing": 1, "summary_empty": 1}
    text = _load_text(summary_file)
    empty = 0 if text.strip() else 1
    return {"summary_missing": 0, "summary_empty": empty}


def _caption_has_error(record: Dict[str, object]) -> bool:
    desc = str(record.get("caption") or record.get("clip_description") or "")
    lower = desc.lower()
    # Treat "[ERROR] No frames sampled" as a stable marker, not a transient error.
    if "[error]" in lower and "no frames sampled" in lower:
        return False
    for token in BAD_TOKENS:
        if token.lower() in lower:
            return True
    return False


def check_semantic(cache_root: Path, indexes_root: Path, video_id: Path) -> Dict[str, int]:
    """Check unified semantic outputs and underlying clip ckpt cache for errors.

    Backward-compatible keys:
      - clip_final_missing: semantic captions missing
      - clip_final_error:   semantic captions contain error markers
      - clip_ckpt_error:    clip caption checkpoints contain error markers
    """
    clip_ckpt_dir = cache_root / "captions_ckpt" / video_id
    sem_dir = indexes_root / video_id
    sem_caps = sem_dir / "semantic_captions.json"

    final_missing = 0 if sem_caps.exists() else 1
    final_error = 0
    if sem_caps.exists():
        try:
            data = json.loads(sem_caps.read_text(encoding="utf-8"))
            for key, value in data.items():
                if key == "subject_registry":
                    continue
                if isinstance(value, dict):
                    if _caption_has_error(value):
                        final_error = 1
                        break
                    cap = str((value or {}).get("caption") or "")
                    for token in BAD_TOKENS:
                        if token.lower() in cap.lower():
                            final_error = 1
                            break
                if final_error:
                    break
        except Exception:
            final_error = 1

    ckpt_error = 0
    if clip_ckpt_dir.exists():
        for json_path in clip_ckpt_dir.rglob("*.json"):
            text = _load_text(json_path)
            try:
                data = json.loads(text)
            except Exception:
                data = {"clip_description": text}
            if isinstance(data, dict) and _caption_has_error(data):
                ckpt_error = 1
                break

    return {
        "clip_final_missing": final_missing,
        "clip_final_error": final_error,
        "clip_ckpt_error": ckpt_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check cached summary/semantic artifacts for a given video_id.")
    parser.add_argument("--summaries-root", required=True, help="Root directory for summaries")
    parser.add_argument("--cache-dir", required=True, help="Cache directory (contains captions_ckpt)")
    parser.add_argument("--indexes-root", required=True, help="Indexes root (contains semantic/<video_id>)")
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g., BENCHMARK/slug)")
    args = parser.parse_args()

    summaries_root = Path(args.summaries_root)
    cache_root = Path(args.cache_dir)
    indexes_root = Path(args.indexes_root) / "semantic"
    video_id = Path(args.video_id)

    summary_status = check_summary(summaries_root, video_id)
    semantic_status = check_semantic(cache_root, indexes_root, video_id)
    status = {**summary_status, **semantic_status}
    print(" ".join(f"{k}={v}" for k, v in status.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

