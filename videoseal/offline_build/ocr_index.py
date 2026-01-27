from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from videoseal.utils.RAG.build_srt_captions_json import srt_to_captions_json
from videoseal.utils.RAG.rag_index_embed import build_embedding_index_npy


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _npy_index_complete(vectors_path: Path, doc_ids_path: Path, meta_path: Path, norms_path: Path) -> bool:
    return vectors_path.exists() and doc_ids_path.exists() and meta_path.exists() and norms_path.exists()


def build_ocr_index(
    *,
    video_id: str,
    srt_path: str,
    embedding_model: str,
    out_dir: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    vid = str(video_id).strip()
    if not vid:
        raise ValueError("video_id is required")

    out_dir_vid = Path(out_dir) / vid
    cap_path = out_dir_vid / "ocr_captions.json"
    ocr_vec_path = out_dir_vid / "ocr_vectors.npy"
    ocr_norms_path = out_dir_vid / "ocr_norms.npy"
    ocr_ids_path = out_dir_vid / "ocr_doc_ids.txt"
    ocr_meta_path = out_dir_vid / "ocr_meta.json"

    if _npy_index_complete(ocr_vec_path, ocr_ids_path, ocr_meta_path, ocr_norms_path) and not overwrite:
        return {
            "video_id": vid,
            "index_dir": out_dir_vid.as_posix(),
            "vectors_path": ocr_vec_path.as_posix(),
            "norms_path": ocr_norms_path.as_posix(),
            "doc_ids_path": ocr_ids_path.as_posix(),
            "meta_path": ocr_meta_path.as_posix(),
            "captions_path": cap_path.as_posix(),
            "docs": None,
            "schema_version": "0.3.1",
            "_cache_hit": True,
        }

    _ensure_dir(out_dir_vid)
    fused_src = Path(srt_path).with_suffix(".ocr_captions.json")
    captions = None
    if fused_src.exists():
        cap_path.write_text(fused_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        captions = srt_to_captions_json(srt_path)
        cap_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

    skip_embed = _env_flag("SKIP_OCR_EMBED") or _env_flag("OFFLINE_SKIP_OCR_EMBED") or _env_flag("SKIP_STAGE_EMBED")
    if not skip_embed:
        build_embedding_index_npy(cap_path, out_dir_vid, prefix="ocr", model=embedding_model)

    docs = None
    try:
        if captions is not None:
            docs = sum(1 for k in captions.keys() if k != "subject_registry")
    except Exception:
        docs = None

    return {
        "video_id": vid,
        "index_dir": out_dir_vid.as_posix(),
        "vectors_path": ocr_vec_path.as_posix(),
        "norms_path": ocr_norms_path.as_posix(),
        "doc_ids_path": ocr_ids_path.as_posix(),
        "meta_path": ocr_meta_path.as_posix(),
        "captions_path": cap_path.as_posix(),
        "docs": docs,
        "schema_version": "0.3.1",
        "_cache_hit": False,
    }

