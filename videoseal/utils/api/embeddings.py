from __future__ import annotations

"""Embedding utilities (OpenAI-compatible endpoint).

Required env vars:
  - EMBEDDING_API_BASE
  - EMBEDDING_API_KEY

Optional:
  - EMBEDDING_MODEL (default: text-embedding-3-large)
  - OPENAI_ORG / OPENAI_ORGANIZATION
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large"


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        base = os.getenv("EMBEDDING_API_BASE")
        key = os.getenv("EMBEDDING_API_KEY")
        if not base or not str(base).strip():
            raise RuntimeError("Missing required env var: EMBEDDING_API_BASE")
        if not key or not str(key).strip():
            raise RuntimeError("Missing required env var: EMBEDDING_API_KEY")
        org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
        return cls(api_key=str(key).strip(), base_url=str(base).strip(), organization=org)


def _mk_openai_client(cfg: OpenAIConfig):
    from openai import OpenAI  # type: ignore

    default_headers = None
    if cfg.base_url and "dashscope.aliyuncs.com" in cfg.base_url:
        default_headers = {
            "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
        }
    kwargs = {
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "organization": cfg.organization,
    }
    if default_headers:
        kwargs["default_headers"] = default_headers
    return OpenAI(**kwargs)  # type: ignore[arg-type]


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    cfg: Optional[OpenAIConfig] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 512,
    progress: bool = False,
) -> List[List[float]]:
    cfg = cfg or OpenAIConfig.from_env()
    if base_url is not None:
        cfg.base_url = base_url
    if api_key is not None:
        cfg.api_key = api_key
    client = _mk_openai_client(cfg)

    indexed = [(i, (t or "").strip()) for i, t in enumerate(texts)]
    non_empty = [(i, t) for i, t in indexed if t]
    if not non_empty:
        return [[] for _ in texts]

    batch_size = int(os.getenv("EMBED_BATCH_SIZE", str(batch_size)))
    timeout_s = float(os.getenv("EMBED_TIMEOUT", "60"))

    results: List[Tuple[int, List[float]]] = []
    rng = range(0, len(non_empty), batch_size)
    iterator = rng
    if progress:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(rng, desc=f"Embedding {len(non_empty)} texts", unit="batch")

    for start in iterator:
        chunk = non_empty[start : start + batch_size]
        idxs, chunk_texts = zip(*chunk)
        resp = client.embeddings.create(
            model=model,
            input=list(chunk_texts),
            encoding_format="float",
            timeout=timeout_s,
        )
        if not getattr(resp, "data", None):
            raise RuntimeError("Embedding response missing 'data'")
        vectors = [d.embedding for d in resp.data]
        results.extend(list(zip(idxs, vectors)))

    dim = len(results[0][1]) if results else 0
    out: List[List[float]] = [[] for _ in texts]
    for i, vec in results:
        out[i] = vec
    for i in range(len(out)):
        if not out[i]:
            out[i] = [0.0] * dim
    return out


def embed_srt(
    srt_path: str,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    jsonl_out: Optional[str] = None,
    npy_out: Optional[str] = None,
    cfg: Optional[OpenAIConfig] = None,
) -> Tuple[Optional[str], Optional[str]]:
    import json

    import pysrt

    subs = pysrt.open(srt_path, encoding="utf-8")
    items = []
    for i, sub in enumerate(subs, 1):
        items.append(
            {
                "index": i,
                "text": (sub.text or "").strip(),
                "start": str(sub.start),
                "end": str(sub.end),
                "start_ms": int(sub.start.ordinal),
                "end_ms": int(sub.end.ordinal),
            }
        )
    texts = [it["text"] for it in items]
    vectors = embed_texts(texts, model=model, cfg=cfg)

    jpath = None
    if jsonl_out:
        os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for it, vec in zip(items, vectors):
                rec = dict(it)
                rec["embedding"] = vec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jpath = jsonl_out

    npath = None
    if npy_out:
        os.makedirs(os.path.dirname(npy_out), exist_ok=True)
        arr = np.array(vectors, dtype=np.float32)
        np.save(npy_out, arr)
        npath = npy_out

    return jpath, npath

