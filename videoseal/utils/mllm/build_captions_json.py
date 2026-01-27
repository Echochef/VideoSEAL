from __future__ import annotations

import argparse
import decimal
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .video_io import get_video_duration_sec, chunk_ranges
from ..api.mllm import MLLMClient
from ..video.time import sec_to_hhmmss
from videoseal.prompts.caption_prompts import default_caption_prompt
from .captioner import merge_subject_registry, _extract_json_object
from ..video.frames import sample_uniform_frames
from ..env_paths import get_cache_root, get_frames_root


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


_BAD_CAPTION_TOKENS = (
    "httperror",
    "client error",
    "forbidden",
    "too many requests",
    "[error]",
)


def _slugify(stem: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()


def _as_text_list(val: Any) -> List[str]:
    out: List[str] = []
    if isinstance(val, list):
        for it in val:
            if isinstance(it, str):
                t = it.strip()
                if t and t not in out:
                    out.append(t)
    return out


def _normalize_entities(cd: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize multiple possible entity sources into a flat list of {type,text} dicts."""
    ents: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    def _add(ty: str, text: str) -> None:
        t = str(text or "").strip()
        if not t:
            return
        key = (ty, t)
        if key in seen:
            return
        seen.add(key)
        ents.append({"type": ty, "text": t})

    # 1) direct entities
    for it in (cd.get("entities") or []):
        if isinstance(it, dict):
            _add(str(it.get("type") or "").strip() or "entity", str(it.get("text") or it.get("name") or ""))
        elif isinstance(it, str):
            _add("entity", it)

    # 2) objects
    for obj in (cd.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        _add("object", str(obj.get("name") or ""))
        for a in (obj.get("attributes") or []):
            _add("attribute", str(a or ""))

    # 3) actions (subject/verb/object)
    for ac in (cd.get("actions") or []):
        if not isinstance(ac, dict):
            continue
        _add("subject", str(ac.get("subject") or ""))
        _add("action", str(ac.get("verb") or ""))
        _add("object", str(ac.get("object") or ""))

    # 4) scene fields
    scene = cd.get("scene")
    if isinstance(scene, dict):
        _add("scene", str(scene.get("location") or ""))
        _add("scene", str(scene.get("lighting") or ""))
        _add("scene", str(scene.get("camera") or ""))

    # 5) text_in_frame, keywords
    for t in _as_text_list(cd.get("text_in_frame")):
        _add("text", t)
    for t in _as_text_list(cd.get("keywords")):
        _add("keyword", t)

    # 6) subject_registry names (as person/subject)
    registry = cd.get("subject_registry")
    if isinstance(registry, dict):
        for name in registry.keys():
            _add("person", str(name or ""))

    return ents


def _is_valid_cached_caption(rec: Any, *, retry_error_caps: bool) -> bool:
    """Return True if a cached record is healthy enough to reuse."""
    if not isinstance(rec, dict):
        return False
    desc = str(rec.get("clip_description") or rec.get("caption") or "").strip()
    if not desc:
        return False
    desc_l = desc.lower()

    # Force retry when the cache contains an explicit error marker (opt-in).
    if retry_error_caps and "[error]" in desc_l:
        return False

    # Treat "[ERROR] No frames sampled" as a stable cache marker.
    if "[error]" in desc_l and "no frames sampled" in desc_l:
        return True

    return not any(t in desc_l for t in _BAD_CAPTION_TOKENS)


def build_captions_for_video(
    video_path: str,
    chunk_seconds: int = 120,
    fps_sample: float = 2.0,
    max_frames: int = 32,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    workers: int = 4,
    ckpt_dir: str | None = None,
    skip_existing: bool = True,
    *,
    video_id_override: Optional[str] = None,
    frames_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end: split video into fixed-length chunks, sample frames, call MLLM
    to generate visual-only captions per chunk, and return a dict similar to DVD's
    captions.json format:

    {
      "<startSec_endSec>": {"caption": "...", "entities": [...]},
      ...,
      "subject_registry": {...}
    }
    """
    duration = get_video_duration_sec(video_path)
    chunks = chunk_ranges(duration, chunk_seconds)
    client = MLLMClient(base_url=api_base, api_key=api_key, model=model)
    retry_error_caps = _env_true("RETRY_ERROR_CAPTIONS", default=False)

    # Effective video_id (may contain benchmark prefix).
    video_id_eff = (video_id_override or _slugify(Path(video_path).stem)).strip()

    # Frames root (allow override to control benchmark prefix).
    frames_dir = Path(frames_root) if frames_root is not None else (get_frames_root() / video_id_eff)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Per-window checkpoint directory (stable, not date-based).
    if ckpt_dir is None:
        base_cache = get_cache_root()
        vp = Path(video_path)
        # Group under captions_ckpt/<video_id> when possible; otherwise, keep legacy naming.
        if video_id_eff:
            ckpt_dir = (base_cache / "captions_ckpt" / video_id_eff).as_posix()
        else:
            ckpt_dir = (base_cache / f"{vp.stem}_captions_ckpt").as_posix()
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {}
    all_entities: List[List[dict]] = []
    # Stats for visibility: how many segments are reused vs recomputed
    stats = {
        "total": 0,
        "cached": 0,
        "recomputed": 0,
        "errors": 0,
        "no_frames": 0,
    }

    def _load_ckpt(key: str) -> Dict[str, Any] | None:
        p = ckpt_path / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to parse ckpt {p}: {type(exc).__name__}: {exc} — recomputing", flush=True)
            return None

    def _save_ckpt(key: str, data: Dict[str, Any]) -> None:
        p = ckpt_path / f"{key}.json"
        safe_payload = _json_safe(data)
        p.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _caption_one_window(s: float, e: float) -> Tuple[Dict[str, Any], str]:
        start_hms = sec_to_hhmmss(s)
        end_hms = sec_to_hhmmss(e)

        frame_sampling_error: Optional[str] = None
        try:
            frame_paths, ts = sample_uniform_frames(
                video_path,
                s,
                e,
                fps=fps_sample,
                max_frames=max_frames,
                output_dir=str(frames_dir),
            )
        except Exception as exc:
            frame_sampling_error = f"{type(exc).__name__}: {exc}"
            frame_paths, ts = [], []
            print(
                f"[WARN] Frame sampling failed; skip MLLM call. window={s:.3f}-{e:.3f} err={frame_sampling_error}",
                flush=True,
            )

        if not frame_paths:
            print(f"[WARN] No frames sampled; skip MLLM call. window={s:.3f}-{e:.3f}", flush=True)
            caption_dict: Dict[str, Any] = {
                "clip_start_time": start_hms,
                "clip_end_time": end_hms,
                "clip_description": "[ERROR] No frames sampled",
                "caption": "",
                "entities": [],
            }
            if frame_sampling_error:
                caption_dict["_frame_sampling_error"] = frame_sampling_error
            return caption_dict, "no_frames"

        try:
            text = client.generate_images_paths(
                frame_paths,
                default_caption_prompt(start_hms, end_hms),
                response_json=True,
                timestamps=ts,
            )
            caption_dict = _extract_json_object(text)
            if caption_dict is None:
                caption_dict = {
                    "clip_start_time": start_hms,
                    "clip_end_time": end_hms,
                    "clip_description": text,
                    "entities": [],
                }
            mode = "recomputed"
        except Exception as exc:
            caption_dict = {
                "clip_start_time": start_hms,
                "clip_end_time": end_hms,
                "clip_description": f"[ERROR] {type(exc).__name__}: {exc}",
                "entities": [],
            }
            mode = "error"

        usage = client.get_last_usage()
        if usage:
            caption_dict["_usage"] = _json_safe(usage)
        return caption_dict, mode

    def _process_one(s: float, e: float) -> Tuple[str, Dict[str, Any], str]:
        key = f"{int(s)}_{int(e)}"

        caption_dict: Dict[str, Any]
        if skip_existing:
            cached = _load_ckpt(key)
            if _is_valid_cached_caption(cached, retry_error_caps=retry_error_caps):
                assert isinstance(cached, dict)
                caption_dict = cached
                mode = "cached"
            else:
                caption_dict, mode = _caption_one_window(s, e)
                _save_ckpt(key, caption_dict)
        else:
            caption_dict, mode = _caption_one_window(s, e)
            _save_ckpt(key, caption_dict)

        entities = _normalize_entities(caption_dict)
        payload = {
            "caption": caption_dict.get("clip_description", ""),
            "entities": entities,
            "_src_entities": (caption_dict.get("entities") or []),
            # Expose original structured fields to avoid information loss
            "scene": caption_dict.get("scene") or None,
            "objects": caption_dict.get("objects") or [],
            "actions": caption_dict.get("actions") or [],
            "text_in_frame": caption_dict.get("text_in_frame") or [],
            "keywords": caption_dict.get("keywords") or [],
        }
        return key, payload, mode

    # progress bar (tqdm if available)
    # progress bar / fallback counter
    _use_tqdm = False
    processed = 0
    step_print = max(1, len(chunks) // 20)  # fallback print 5% steps
    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=len(chunks), desc=f"Captioning {Path(video_path).name}")
        _use_tqdm = True
    except Exception:
        pbar = None
        print("[info] tqdm not installed; captioning without progress bar")

    def _tick():
        nonlocal processed
        if _use_tqdm and pbar is not None:
            pbar.update(1)
        else:
            processed += 1
            if processed % step_print == 0 or processed == len(chunks):
                print(f"[caption] {processed}/{len(chunks)}", flush=True)

    def _close():
        if _use_tqdm and pbar is not None:
            pbar.close()

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, s, e): (s, e) for (s, e) in chunks}
            for fut in as_completed(futures):
                key, payload, mode = fut.result()
                result[key] = payload
                all_entities.append(payload.get("_src_entities") or [])
                # update stats in main thread
                stats["total"] += 1
                if mode == "cached":
                    stats["cached"] += 1
                elif mode == "recomputed":
                    stats["recomputed"] += 1
                elif mode == "error":
                    stats["errors"] += 1
                elif mode == "no_frames":
                    stats["no_frames"] += 1
                _tick()
    else:
        for (s, e) in chunks:
            key, payload, mode = _process_one(s, e)
            result[key] = payload
            all_entities.append(payload.get("_src_entities") or [])
            stats["total"] += 1
            if mode == "cached":
                stats["cached"] += 1
            elif mode == "recomputed":
                stats["recomputed"] += 1
            elif mode == "error":
                stats["errors"] += 1
            elif mode == "no_frames":
                stats["no_frames"] += 1
            _tick()
    _close()

    print(
        f"[CAPTION] segments total={stats['total']} cached={stats['cached']} "
        f"recomputed={stats['recomputed']} errors={stats['errors']} no_frames={stats['no_frames']}",
        flush=True,
    )

    result["subject_registry"] = merge_subject_registry(all_entities)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-clip captions (visual-only) JSON for a video")
    parser.add_argument("--video", required=True, help="Absolute path to video file")
    parser.add_argument("--out", required=True, help="Output captions.json path")
    parser.add_argument("--chunk-seconds", type=int, default=120, help="Chunk length in seconds")
    parser.add_argument("--fps-sample", type=float, default=2.0, help="Sampling FPS for frames")
    parser.add_argument("--max-frames", type=int, default=32, help="Max frames per chunk")
    parser.add_argument("--model", type=str, default=None, help="Override MLLM model name")
    parser.add_argument("--api-base", type=str, default=None, help="Override MLLM base URL")
    parser.add_argument("--api-key", type=str, default=None, help="Override MLLM API key")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (threads). Use 1 for sequential.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint folder for per-window JSON (default: <cache>/captions_ckpt/<video_id>)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute even if checkpoint files exist.")
    parser.add_argument("--lang", type=str, default=None, choices=["zh", "en"], help="Prompt language override (writes RAG_PROMPT_LANG env).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If specified, write to env so prompt builders can pick it up.
    if args.lang:
        os.environ["RAG_PROMPT_LANG"] = args.lang

    data = build_captions_for_video(
        video_path=args.video,
        chunk_seconds=args.chunk_seconds,
        fps_sample=args.fps_sample,
        max_frames=args.max_frames,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        workers=args.workers,
        ckpt_dir=args.ckpt_dir,
        skip_existing=(not args.no_skip_existing),
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved captions json to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
