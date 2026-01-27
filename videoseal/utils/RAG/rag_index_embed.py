from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..api.embeddings import DEFAULT_EMBED_MODEL, embed_texts


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _extract_entity_texts(val: Any) -> List[str]:
    out: List[str] = []
    if isinstance(val, list):
        for it in val:
            t = None
            if isinstance(it, dict):
                t = it.get("text") or it.get("name") or None
            elif isinstance(it, str):
                t = it
            if t:
                t = str(t).strip()
                if t and t not in out:
                    out.append(t)
    return out


def build_embedding_index_npy(
    captions_path: Path,
    out_dir: Path,
    *,
    prefix: str,
    model: str | None = None,
) -> Dict[str, str]:
    """Build embedding index and save as NumPy files."""
    data = json.loads(captions_path.read_text(encoding="utf-8"))
    doc_ids: List[str] = []
    texts: List[str] = []
    for k, v in data.items():
        if k == "subject_registry":
            continue
        rec = v or {}
        cap = str(rec.get("caption") or "").strip()
        ents = _extract_entity_texts(rec.get("entities"))
        cap_ext = (cap + " " + " ".join(ents)).strip() if ents else cap
        if not cap_ext:
            continue
        doc_ids.append(k)
        texts.append(cap_ext)

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_out = out_dir / f"{prefix}_vectors.npy"
    norm_out = out_dir / f"{prefix}_norms.npy"
    ids_out = out_dir / f"{prefix}_doc_ids.txt"
    meta_out = out_dir / f"{prefix}_meta.json"

    if not doc_ids:
        np.save(vec_out, np.asarray([], dtype=np.float32))
        np.save(norm_out, np.asarray([], dtype=np.float32))
        ids_out.write_text("", encoding="utf-8")
        meta_out.write_text(
            json.dumps(
                {
                    "model": model or DEFAULT_EMBED_MODEL,
                    "dimension": 0,
                    "rows": 0,
                    "source_json": captions_path.name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "vectors": vec_out.as_posix(),
            "norms": norm_out.as_posix(),
            "doc_ids": ids_out.as_posix(),
            "meta": meta_out.as_posix(),
        }

    vectors: List[List[float]] = embed_texts(texts, model=(model or DEFAULT_EMBED_MODEL), progress=True)
    norms: List[float] = [_norm(v) for v in vectors]
    dim = len(vectors[0]) if vectors else 0

    np.save(vec_out, np.asarray(vectors, dtype=np.float32))
    np.save(norm_out, np.asarray(norms, dtype=np.float32))
    with ids_out.open("w", encoding="utf-8") as f:
        for did in doc_ids:
            f.write(str(did) + "\n")
    meta_out.write_text(
        json.dumps(
            {
                "model": model or DEFAULT_EMBED_MODEL,
                "dimension": int(dim),
                "rows": int(len(doc_ids)),
                "source_json": captions_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "vectors": vec_out.as_posix(),
        "norms": norm_out.as_posix(),
        "doc_ids": ids_out.as_posix(),
        "meta": meta_out.as_posix(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build embedding index (OpenAI) from captions.json")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", default=None, help="Embedding model name (default: EMBEDDING_MODEL or text-embedding-3-large)")
    args = ap.parse_args()

    info = build_embedding_index_npy(Path(args.captions), Path(args.out_dir), prefix=str(args.prefix), model=args.model)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

