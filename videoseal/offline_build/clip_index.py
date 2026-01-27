from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from videoseal.utils.RAG.rag_index_embed import build_embedding_index_npy
from videoseal.utils.env_paths import get_cache_root, get_frames_root


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _npy_index_complete(vectors_path: Path, doc_ids_path: Path, meta_path: Path, norms_path: Path) -> bool:
    return vectors_path.exists() and doc_ids_path.exists() and meta_path.exists() and norms_path.exists()


def _captions_has_http_error(captions_path: Path) -> bool:
    try:
        text = captions_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return True
    bad = ("httperror", "client error", "forbidden", "too many requests", "[error]")
    return any(t in text for t in bad)


def build_clip_index(
    *,
    video_path: str,
    video_id: str,
    clip_len_sec: float,
    sample_fps: float,
    embedding_model: str,
    out_dir: str,
    max_frames: int = 32,
    workers: int = 4,
    overwrite: bool = False,
) -> Dict[str, Any]:
    vid = str(video_id).strip()
    if not vid:
        raise ValueError("video_id is required")

    out_dir_vid = Path(out_dir) / vid
    cap_path = out_dir_vid / "captions.json"
    clip_vec_path = out_dir_vid / "clip_vectors.npy"
    clip_norms_path = out_dir_vid / "clip_norms.npy"
    clip_ids_path = out_dir_vid / "clip_doc_ids.txt"
    clip_meta_path = out_dir_vid / "clip_meta.json"

    if (
        _npy_index_complete(clip_vec_path, clip_ids_path, clip_meta_path, clip_norms_path)
        and cap_path.exists()
        and not overwrite
        and not _captions_has_http_error(cap_path)
    ):
        return {
            "video_id": vid,
            "index_dir": out_dir_vid.as_posix(),
            "vectors_path": clip_vec_path.as_posix(),
            "norms_path": clip_norms_path.as_posix(),
            "doc_ids_path": clip_ids_path.as_posix(),
            "meta_path": clip_meta_path.as_posix(),
            "captions_path": cap_path.as_posix(),
            "docs": None,
            "schema_version": "0.3.1",
            "_cache_hit": True,
        }

    from videoseal.utils.mllm.build_captions_json import build_captions_for_video

    frames_dir = get_frames_root() / vid
    cache_ckpt_dir = get_cache_root() / "captions_ckpt" / vid
    _ensure_dir(frames_dir)
    _ensure_dir(cache_ckpt_dir)

    data = build_captions_for_video(
        video_path=video_path,
        chunk_seconds=int(clip_len_sec),
        fps_sample=float(sample_fps),
        max_frames=int(max_frames),
        workers=int(workers),
        skip_existing=(not overwrite),
        video_id_override=vid,
        frames_root=str(frames_dir),
        ckpt_dir=str(cache_ckpt_dir),
    )
    _ensure_dir(out_dir_vid)
    cap_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    skip_embed = _env_flag("SKIP_CLIP_EMBED") or _env_flag("OFFLINE_SKIP_CLIP_EMBED") or _env_flag("SKIP_STAGE_EMBED")
    if not skip_embed:
        build_embedding_index_npy(cap_path, out_dir_vid, prefix="clip", model=embedding_model)

    docs = None
    try:
        docs = sum(1 for k in (data or {}).keys() if k != "subject_registry")
    except Exception:
        docs = None

    return {
        "video_id": vid,
        "index_dir": out_dir_vid.as_posix(),
        "vectors_path": clip_vec_path.as_posix(),
        "norms_path": clip_norms_path.as_posix(),
        "doc_ids_path": clip_ids_path.as_posix(),
        "meta_path": clip_meta_path.as_posix(),
        "captions_path": cap_path.as_posix(),
        "docs": docs,
        "schema_version": "0.3.1",
        "_cache_hit": False,
    }

