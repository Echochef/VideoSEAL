from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple, TypeVar

K = TypeVar("K")


def normalize_score_map(scores: Mapping[K, float]) -> Dict[K, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return {k: 1.0 for k in scores.keys()}
    return {k: (float(v) - float(vmin)) / (float(vmax) - float(vmin)) for k, v in scores.items()}


def fuse_weighted_hits(
    embed_hits: Iterable[Tuple[str, float]],
    bm25_hits: Iterable[Tuple[str, float]],
    *,
    w_embed: float,
    w_bm25: float,
    top_k: int,
) -> List[Tuple[str, float]]:
    e_list = list(embed_hits or [])
    b_list = list(bm25_hits or [])
    cand_ids = {d for d, _ in e_list} | {d for d, _ in b_list}
    e_map = {d: float(s) for d, s in e_list}
    b_map = {d: float(s) for d, s in b_list}
    e_norm = normalize_score_map(e_map)
    b_norm = normalize_score_map(b_map)
    comb: List[Tuple[str, float]] = []
    for d in cand_ids:
        se = e_norm.get(d, 0.0)
        sb = b_norm.get(d, 0.0)
        comb.append((d, float(float(w_embed) * float(se) + float(w_bm25) * float(sb))))
    comb.sort(key=lambda x: x[1], reverse=True)
    return comb[: max(int(top_k), 0)]

