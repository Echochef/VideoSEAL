from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    s = (text or "").lower()
    return re.findall(r"[\\u4e00-\\u9fff]|[a-z0-9]+", s)


def _load_captions(index_dir: Path) -> Dict[str, Dict]:
    caps_path = Path(index_dir) / "semantic_captions.json"
    if not caps_path.exists():
        return {}
    try:
        return json.loads(caps_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bm25_idf(N: int, df: int) -> float:
    return math.log(1.0 + ((N - df + 0.5) / (df + 0.5))) if df > 0 else 0.0


def _bm25_score_doc(
    query_terms: List[str],
    term_freq: Dict[str, int],
    dl: int,
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    idf_map: Dict[str, float] | None = None,
) -> float:
    score = 0.0
    for t in query_terms:
        tf = term_freq.get(t, 0)
        if tf <= 0:
            continue
        idf = idf_map.get(t, 0.0) if idf_map else 0.0
        denom = tf + k1 * (1.0 - b + b * (dl / (avgdl or 1.0)))
        score += idf * ((tf * (k1 + 1.0)) / (denom or 1.0))
    return score


def query_bm25(index_dir: Path | str, query_text: str, topk: int = 50) -> List[Tuple[str, float]]:
    d = Path(index_dir)
    items = _load_captions(d)

    doc_ids: List[str] = []
    doc_terms: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = {}

    for k, meta in items.items():
        if k == "subject_registry":
            continue
        cap = ""
        if isinstance(meta, dict):
            cap = str(meta.get("caption") or meta.get("clip_caption") or meta.get("ocr_text") or "").strip()
        toks = _tokenize(cap)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
        doc_ids.append(k)
        doc_terms.append(tf)
        doc_lens.append(len(toks))

    N = len(doc_ids)
    if N == 0:
        return []
    avgdl = (sum(doc_lens) / float(N)) if N > 0 else 0.0
    q_terms = _tokenize(query_text)
    q_unique = list(dict.fromkeys(q_terms))
    idf_map: Dict[str, float] = {t: _bm25_idf(N, df.get(t, 0)) for t in q_unique}

    scores: List[Tuple[str, float]] = []
    for i, tf in enumerate(doc_terms):
        s = _bm25_score_doc(q_unique, tf, doc_lens[i], avgdl, idf_map=idf_map)
        if s > 0.0:
            scores.append((doc_ids[i], float(s)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

