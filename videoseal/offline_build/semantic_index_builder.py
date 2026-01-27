from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from videoseal.utils.RAG.rag_index_embed import build_embedding_index_npy


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_key(key: str) -> Tuple[int, int]:
    try:
        start_s, end_s = key.split("_", 1)
        return int(float(start_s)), int(float(end_s))
    except Exception:
        return 0, 0


def _collect_ocr_segments(data: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    segments: List[Tuple[int, int, str]] = []
    for key, payload in data.items():
        if key == "subject_registry":
            continue
        start_sec, end_sec = _parse_key(key)
        text = str((payload or {}).get("caption") or "").strip()
        if not text:
            continue
        segments.append((start_sec, end_sec, text))
    segments.sort(key=lambda x: (x[0], x[1]))
    return segments


def _merge_texts(
    clip_caps: Dict[str, Any],
    ocr_segments: Sequence[Tuple[int, int, str]],
    *,
    video_id: str,
) -> Tuple[Dict[str, Any], List[str]]:
    merged: Dict[str, Any] = {}
    text_lines: List[str] = []
    subject_registry = clip_caps.get("subject_registry") if isinstance(clip_caps, dict) else {}
    for key, payload in clip_caps.items():
        if key == "subject_registry":
            continue
        start_sec, end_sec = _parse_key(key)
        clip_caption = str((payload or {}).get("caption") or "").strip()
        clip_entities = payload.get("entities") if isinstance(payload, dict) else None
        overlapping: List[str] = []
        for osc, oec, otext in ocr_segments:
            if oec < start_sec or osc > end_sec:
                continue
            if otext and otext not in overlapping:
                overlapping.append(otext)
        sections: List[str] = []
        if clip_caption:
            sections.append(f"Visual: {clip_caption}")
        if overlapping:
            sections.append(f"Dialogue: {' '.join(overlapping)}")
        combined = "\n".join(sections).strip()
        if not combined:
            combined = clip_caption or ("Dialogue: " + " ".join(overlapping) if overlapping else "")
        merged[key] = {
            "caption": combined,
            "clip_caption": clip_caption,
            "ocr_text": " ".join(overlapping),
            "entities": clip_entities if isinstance(clip_entities, list) else [],
            "source": {
                "clip": bool(clip_caption),
                "ocr": bool(overlapping),
            },
        }
        if combined:
            doc_line = f"[{video_id}] {start_sec}-{end_sec}s\n{combined}\n"
            text_lines.append(doc_line)
    merged["subject_registry"] = subject_registry or {}
    return merged, text_lines


def _write_semantic_assets(output_dir: Path, captions: Dict[str, Any], text_lines: Iterable[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    captions_path = output_dir / "semantic_captions.json"
    captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

    text_path = output_dir / "semantic_segments.txt"
    text_content = "\n".join(text_lines)
    text_path.write_text(text_content, encoding="utf-8")
    return captions_path


def _rebuild_embeddings(captions_path: Path, output_dir: Path, *, model: str) -> Dict[str, str]:
    info = build_embedding_index_npy(captions_path, output_dir, prefix="semantic", model=model)

    vectors_path = Path(info["vectors"])
    norms_path = Path(info["norms"])
    doc_ids_path = Path(info["doc_ids"])
    meta_path = Path(info["meta"])

    vectors = np.load(vectors_path, mmap_mode="r")
    norms = np.load(norms_path, mmap_mode="r") if norms_path.exists() else np.linalg.norm(vectors, axis=1)
    doc_ids = [ln.strip() for ln in doc_ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embed_model = meta.get("model", model)
        dimension = int(meta.get("dimension", vectors.shape[1] if vectors.ndim == 2 else 0))
    except Exception:
        embed_model = model
        dimension = vectors.shape[1] if vectors.ndim == 2 else 0

    json_index = {
        "doc_ids": doc_ids,
        "vectors": vectors.tolist(),
        "norms": norms.tolist(),
        "model": embed_model,
        "dimension": dimension,
    }
    (output_dir / "semantic_index.json").write_text(json.dumps(json_index, ensure_ascii=False), encoding="utf-8")
    return info


def _discover_video_ids(clip_root: Path) -> List[str]:
    video_ids: List[str] = []
    for path in clip_root.rglob("captions.json"):
        rel = path.parent.relative_to(clip_root).as_posix()
        if rel and rel not in video_ids:
            video_ids.append(rel)
    video_ids.sort()
    return video_ids


def _build_one_semantic(
    video_id: str,
    *,
    clip_root: Path,
    ocr_root: Path,
    semantic_root: Path,
    model: str,
    overwrite: bool,
) -> Tuple[str, Dict[str, str]]:
    clip_dir = clip_root / video_id
    ocr_dir = ocr_root / video_id
    if not clip_dir.exists():
        print(f"[SKIP] clip captions missing for {video_id}: {clip_dir}", flush=True)
        return video_id, {}
    clip_captions_path = clip_dir / "captions.json"
    try:
        clip_data = _load_json(clip_captions_path)
    except Exception as exc:
        print(f"[SKIP] failed to load clip captions for {video_id}: {exc}", flush=True)
        return video_id, {}

    ocr_captions_path = ocr_dir / "ocr_captions.json"
    ocr_data: Dict[str, Any] = {}
    if ocr_captions_path.exists():
        try:
            ocr_data = _load_json(ocr_captions_path)
        except Exception as exc:
            print(f"[WARN] failed to load OCR captions for {video_id}: {exc}", flush=True)

    output_dir = semantic_root / video_id
    semantic_vec_path = output_dir / "semantic_vectors.npy"
    semantic_caps_path = output_dir / "semantic_captions.json"
    semantic_txt_path = output_dir / "semantic_segments.txt"

    skip_semantic_env = (os.getenv("SKIP_SEMANTIC_EMBED") or os.getenv("SKIP_STAGE_EMBED") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if not overwrite and semantic_vec_path.exists() and semantic_caps_path.exists() and semantic_txt_path.exists():
        print(f"[SKIP] semantic embeddings already exist for {video_id} (use overwrite to rebuild)", flush=True)
        return video_id, {}

    ocr_segments = _collect_ocr_segments(ocr_data) if ocr_data else []
    merged_caps, text_lines = _merge_texts(clip_data, ocr_segments, video_id=video_id)
    captions_path = _write_semantic_assets(output_dir, merged_caps, text_lines)

    if skip_semantic_env:
        print(f"[SKIP] semantic embedding disabled by env for {video_id} (captions only, no vectors)", flush=True)
        return video_id, {}
    if not overwrite and semantic_vec_path.exists():
        print(f"[OK] semantic captions restored for {video_id} (kept existing embeddings) -> {output_dir}", flush=True)
        return video_id, {}

    embed_info = _rebuild_embeddings(captions_path, output_dir, model=model)
    print(f"[OK] semantic index built for {video_id} -> {output_dir}", flush=True)
    return video_id, embed_info


def build_semantic_indices(
    video_ids: Sequence[str],
    *,
    clip_root: Path,
    ocr_root: Path,
    semantic_root: Path,
    model: str,
    overwrite: bool = False,
    workers: int | None = None,
    thread: bool = False,
) -> List[Tuple[str, Dict[str, str]]]:
    if not video_ids:
        return []

    if not workers or workers <= 1:
        results: List[Tuple[str, Dict[str, str]]] = []
        for vid in video_ids:
            results.append(
                _build_one_semantic(
                    vid,
                    clip_root=clip_root,
                    ocr_root=ocr_root,
                    semantic_root=semantic_root,
                    model=model,
                    overwrite=overwrite,
                )
            )
        return results

    Exec = ThreadPoolExecutor if thread else ProcessPoolExecutor
    workers = max(1, int(workers))
    results: List[Tuple[str, Dict[str, str]]] = []
    with Exec(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _build_one_semantic,
                vid,
                clip_root=clip_root,
                ocr_root=ocr_root,
                semantic_root=semantic_root,
                model=model,
                overwrite=overwrite,
            )
            for vid in video_ids
        ]
        for fut in as_completed(futs):
            try:
                vid, info = fut.result()
                results.append((vid, info))
            except Exception as exc:  # pragma: no cover
                print(f"[ERROR] semantic build failed in worker: {exc}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified semantic embeddings by merging clip captions and OCR text.")
    ap.add_argument("--video-id", dest="video_ids", action="append", help="Video ID relative to clip/OCR root (repeatable).")
    ap.add_argument("--clip-root", default="data/cache/semantic_inputs/clip", help="Root directory containing clip captions.")
    ap.add_argument("--ocr-root", default="data/cache/semantic_inputs/ocr", help="Root directory containing OCR captions.")
    ap.add_argument("--semantic-root", default="data/indexes/semantic", help="Output root for semantic indexes.")
    ap.add_argument("--workers", type=int, default=None, help="Parallel workers over video_ids (default: sequential).")
    ap.add_argument("--thread", action="store_true", help="Use thread pool when workers>1.")
    ap.add_argument("--embed-model", default=None, help="Embedding model name (default: EMBEDDING_MODEL or text-embedding-3-large).")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if semantic artifacts already exist.")
    args = ap.parse_args()

    clip_root = Path(args.clip_root).resolve()
    ocr_root = Path(args.ocr_root).resolve()
    semantic_root = Path(args.semantic_root).resolve()

    if args.video_ids:
        video_ids = [vid.strip("/").strip() for vid in args.video_ids if vid]
    else:
        video_ids = _discover_video_ids(clip_root)
        if not video_ids:
            raise SystemExit(f"No clip captions found under {clip_root}")

    embed_model = args.embed_model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large"

    build_semantic_indices(
        video_ids,
        clip_root=clip_root,
        ocr_root=ocr_root,
        semantic_root=semantic_root,
        model=embed_model,
        overwrite=bool(args.overwrite),
        workers=args.workers,
        thread=bool(args.thread),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

