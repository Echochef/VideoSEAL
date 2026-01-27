from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from videoseal.prompts.summary_prompts import (
    build_chunk_scene_fusion_prompt_en,
    build_full_story_prompt,
    build_global_storyline_prompt_en,
)
from videoseal.utils.agent.env import env_float, env_int, env_flag
from videoseal.utils.api.mllm import MLLMClient
from videoseal.utils.video.time import sec_to_hhmmss


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_summary_client(model: Optional[str]) -> MLLMClient:
    """Build SUMMARY client; fallback to MLLM_* when SUMMARY_* missing."""
    s_base = (os.getenv("SUMMARY_API_BASE") or "").strip()
    s_key = (os.getenv("SUMMARY_API_KEY") or "").strip()
    s_model = (os.getenv("SUMMARY_MODEL") or "").strip()
    s_backend = (os.getenv("SUMMARY_BACKEND") or "").strip()

    if not (s_base and s_key and s_model and s_backend):
        m_base = (os.getenv("MLLM_API_BASE") or "").strip()
        m_key = (os.getenv("MLLM_API_KEY") or "").strip()
        m_model = (os.getenv("MLLM_MODEL") or "").strip()
        m_backend = (os.getenv("MLLM_BACKEND") or "openai").strip()
        if not (m_base and m_key and m_model and m_backend):
            raise RuntimeError("Missing SUMMARY_* and MLLM_* env vars for summary generation.")
        cli = MLLMClient(base_url=m_base, api_key=m_key, model=m_model, backend=m_backend)
    else:
        cli = MLLMClient(base_url=s_base, api_key=s_key, model=s_model, backend=s_backend)

    if model:
        cli.model = model
    return cli


def build_global_summary(
    *,
    video_id: str,
    indexes_root: Path,
    summaries_root: Path,
    cache_root: Optional[Path] = None,
    model: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    summ_dir = summaries_root / video_id
    _ensure_dir(summ_dir)
    out_txt = summ_dir / "full_story.txt"
    if out_txt.exists() and not overwrite:
        return out_txt

    semantic_caps = _load_json(indexes_root / "semantic" / video_id / "semantic_captions.json")
    clip_caps = None
    ocr_caps = None
    if semantic_caps is None and cache_root is not None:
        clip_caps = _load_json(cache_root / "semantic_inputs" / "clip" / video_id / "captions.json")
        ocr_caps = _load_json(cache_root / "semantic_inputs" / "ocr" / video_id / "ocr_captions.json")

    if semantic_caps is None and not clip_caps and not ocr_caps:
        raise FileNotFoundError(f"No captions found for summary (semantic/clip/ocr). video_id={video_id}")

    def _iter_caps(d: Dict[str, Any], source: str):
        for k, v in d.items():
            if k == "subject_registry":
                continue
            try:
                s, e = k.split("_")
                s_sec, e_sec = int(float(s)), int(float(e))
            except Exception:
                continue
            text = str((v or {}).get("caption") or "").strip()
            if text:
                yield (s_sec, e_sec, source, text)

    recs: List[Tuple[int, int, str, str]] = []
    if semantic_caps:
        recs.extend(list(_iter_caps(semantic_caps, "SEMANTIC")))
    else:
        if clip_caps:
            recs.extend(list(_iter_caps(clip_caps, "CLIP")))
        if ocr_caps:
            recs.extend(list(_iter_caps(ocr_caps, "OCR")))
    recs.sort(key=lambda x: (x[0], x[1], x[2]))

    mode = (os.getenv("SUMMARY_TIMELINE_MODE") or "window").lower()
    if mode not in ("window", "raw"):
        mode = "window"

    windows_list: List[Tuple[int, int, str]] = []
    if mode == "raw":
        lines: List[str] = []
        for s_sec, e_sec, _src, text in recs:
            lines.append(f"- [{sec_to_hhmmss(s_sec)}–{sec_to_hhmmss(e_sec)}] {text}")
        timeline = "\n".join(lines)
    else:
        win_sec = env_int("SUMMARY_WINDOW_SECONDS", 60)
        win_max_chars = env_int("SUMMARY_WINDOW_MAX_CHARS", 500)
        if not recs:
            timeline = ""
        else:
            start_min = max(0, min(s for s, _, _, _ in recs))
            end_max = max(e for _, e, _, _ in recs)
            base = (start_min // win_sec) * win_sec
            windows: List[Tuple[int, int]] = []
            cur = base
            while cur < end_max:
                windows.append((cur, min(end_max, cur + win_sec)))
                cur += win_sec

            buckets: List[str] = []
            for wstart, wend in windows:
                texts: List[str] = []
                seen: set[str] = set()
                for s_sec, e_sec, _src, text in recs:
                    if e_sec <= wstart or s_sec >= wend:
                        continue
                    t = (text or "").strip()
                    if not t:
                        continue
                    key = t[:120]
                    if key in seen:
                        continue
                    seen.add(key)
                    texts.append(t)
                if not texts:
                    continue
                merged = " ".join(texts)
                if len(merged) > win_max_chars:
                    merged = merged[: win_max_chars - 6].rstrip() + " …"
                buckets.append(f"- [{sec_to_hhmmss(wstart)}–{sec_to_hhmmss(wend)}] {merged}")
                windows_list.append((wstart, wend, merged))
            timeline = "\n".join(buckets)

    cli = _build_summary_client(model)
    sum_max_tokens = env_int("SUMMARY_MAX_TOKENS", 800)
    sum_temperature = env_float("SUMMARY_TEMPERATURE", 0.0)

    two_stage = env_flag("SUMMARY_TWO_STAGE", default=False)
    if two_stage and windows_list:
        chunk_sec = 10 * 60
        wmin = min(ws for ws, _, _ in windows_list)
        wmax = max(we for _, we, _ in windows_list)
        base = (wmin // chunk_sec) * chunk_sec
        chunks_bounds: List[Tuple[int, int]] = []
        cur = base
        while cur < wmax:
            chunks_bounds.append((cur, min(wmax, cur + chunk_sec)))
            cur += chunk_sec

        def _run_one(cstart: int, cend: int) -> Tuple[Tuple[int, int], str] | None:
            local_notes: List[str] = []
            for ws, we, text in windows_list:
                if we <= cstart or ws >= cend:
                    continue
                local_notes.append(f"[{sec_to_hhmmss(ws)}–{sec_to_hhmmss(we)}] {text}")
            if not local_notes:
                return None
            time_label = f"{sec_to_hhmmss(cstart)}–{sec_to_hhmmss(cend)}"
            prompt_chunk = build_chunk_scene_fusion_prompt_en(time_label, "\n".join(local_notes))
            para = cli.generate_text(prompt_chunk, response_json=False, max_tokens=sum_max_tokens, temperature=sum_temperature) or ""
            para = str(para).strip()
            return ((cstart, cend), para)

        workers = max(1, env_int("SUMMARY_CHUNK_WORKERS", 4))
        chunk_texts: List[Tuple[Tuple[int, int], str]] = []
        if workers == 1:
            for cstart, cend in chunks_bounds:
                res = _run_one(cstart, cend)
                if res is not None:
                    chunk_texts.append(res)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_run_one, cstart, cend): (cstart, cend) for cstart, cend in chunks_bounds}
                for fut in as_completed(futs):
                    res = fut.result()
                    if res is not None:
                        chunk_texts.append(res)

        chunk_texts.sort(key=lambda x: (x[0][0], x[0][1]))
        assembled = []
        for (cs, ce), para in chunk_texts:
            assembled.append(f"[{sec_to_hhmmss(cs)}–{sec_to_hhmmss(ce)}]\n{para}")
        prompt_global = build_global_storyline_prompt_en("\n\n".join(assembled))
        story = cli.generate_text(prompt_global, response_json=False, max_tokens=sum_max_tokens, temperature=sum_temperature) or ""
        story = str(story).strip()
    else:
        prompt = build_full_story_prompt(timeline)
        story = cli.generate_text(prompt, response_json=False, max_tokens=sum_max_tokens, temperature=sum_temperature) or ""
        story = str(story).strip()

    out_txt.write_text(story + "\n", encoding="utf-8")
    return out_txt

