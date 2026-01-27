from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..api.embeddings import DEFAULT_EMBED_MODEL, embed_texts


def _cosine(a: List[float], an: float, b: List[float], bn: float) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    for i in range(n):
        dot += a[i] * b[i]
    return dot / (an * bn) if an and bn else 0.0


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _load_index_npy_from_dir(d: Path) -> Tuple[List[str], np.ndarray, np.ndarray, str]:
    if (d / "semantic_vectors.npy").exists():
        prefix = "semantic"
    elif (d / "clip_vectors.npy").exists():
        prefix = "clip"
    elif (d / "ocr_vectors.npy").exists():
        prefix = "ocr"
    else:
        cand = list(d.glob("*_vectors.npy"))
        if not cand:
            raise FileNotFoundError(f"No *_vectors.npy under {d}")
        prefix = cand[0].name.split("_vectors.npy")[0]

    vec_p = d / f"{prefix}_vectors.npy"
    norms_p = d / f"{prefix}_norms.npy"
    ids_p = d / f"{prefix}_doc_ids.txt"
    meta_p = d / f"{prefix}_meta.json"

    V = np.load(vec_p, mmap_mode="r").astype(np.float32)
    if norms_p.exists():
        N = np.load(norms_p, mmap_mode="r").astype(np.float32)
    else:
        N = np.linalg.norm(V, axis=1).astype(np.float32)
    try:
        ids = [ln.strip() for ln in ids_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        ids = [str(i) for i in range(V.shape[0])]
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        model = str(meta.get("model", DEFAULT_EMBED_MODEL))
    except Exception:
        model = DEFAULT_EMBED_MODEL
    return ids, V, N, model


def load_index(index_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, str]:
    if index_path.is_dir():
        return _load_index_npy_from_dir(index_path)
    if index_path.suffix == ".npy":
        return _load_index_npy_from_dir(index_path.parent)
    if index_path.suffix == ".json":
        if not index_path.exists():
            return _load_index_npy_from_dir(index_path.parent)
        data = json.loads(index_path.read_text(encoding="utf-8"))
        doc_ids: List[str] = list(data.get("doc_ids", []))
        vectors = np.asarray(list(data.get("vectors", [])), dtype=np.float32)
        norms = np.asarray([float(x) for x in data.get("norms", [])], dtype=np.float32)
        model = str(data.get("model", DEFAULT_EMBED_MODEL))
        if vectors.ndim != 2:
            raise ValueError(f"invalid vectors in {index_path}")
        if norms.shape != (vectors.shape[0],):
            norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        return doc_ids, vectors, norms, model
    return _load_index_npy_from_dir(index_path)


def query(index_path: Path, query_text: str, topk: int = 5) -> List[Tuple[str, float]]:
    doc_ids, V, N, model = load_index(index_path)
    if not doc_ids:
        return []
    qv = embed_texts([query_text], model=model)[0]
    q = np.asarray(qv, dtype=np.float32)
    denom = (N * (np.linalg.norm(q) or 1.0)).astype(np.float32)
    denom = np.where(denom == 0, 1.0, denom)
    sim = (V @ q) / denom

    n = int(sim.shape[0])
    try:
        k = int(topk)
    except Exception:
        k = 5
    k = max(1, min(k, n))
    idx = np.argpartition(-sim, k - 1)[:k]
    order = idx[np.argsort(-sim[idx])]
    return [(doc_ids[int(i)], float(sim[int(i)])) for i in order]


def query_json(index_json: Path, query_text: str, topk: int = 5) -> List[Tuple[str, float]]:
    """Legacy helper: query a JSON index (doc_ids/vectors/norms)."""
    data = json.loads(index_json.read_text(encoding="utf-8"))
    doc_ids: List[str] = list(data.get("doc_ids", []))
    vecs: List[List[float]] = list(data.get("vectors", []))
    norms: List[float] = [float(x) for x in data.get("norms", [])]
    model = str(data.get("model", DEFAULT_EMBED_MODEL))
    if not doc_ids:
        return []
    qv = embed_texts([query_text], model=model)[0]
    qn = _norm(qv)
    scores: List[Tuple[str, float]] = []
    for i, dv in enumerate(vecs):
        s = _cosine(qv, qn, dv, norms[i] if i < len(norms) else _norm(dv))
        scores.append((doc_ids[i], s))
    scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, min(int(topk), len(scores)))
    return scores[:k]

