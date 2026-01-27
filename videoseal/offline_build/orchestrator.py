from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from videoseal.offline_build.clip_index import build_clip_index
from videoseal.offline_build.ocr_index import build_ocr_index
from videoseal.offline_build.semantic_index_builder import build_semantic_indices
from videoseal.offline_build.summary_builder import build_global_summary
from videoseal.utils.env_paths import get_cache_root, get_indexes_root, get_summaries_root


def _slugify(stem: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _index_complete(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def run_cmd(args: list[str]) -> str:
    """Run a subprocess and stream stdout live; return full stdout as a string."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        print(line, end="", flush=True)
    ret = proc.wait()
    full = "".join(out_lines)
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(args)}\n{full}")
    return full.strip()


def extract_srt(
    video_path: str,
    out_dir: Path,
    *,
    language: str = "ch",
    mode: str = "auto",
    overwrite: bool = False,
    full_image: bool = False,
) -> Path:
    _ensure_dir(out_dir)

    guess = out_dir / (Path(video_path).stem + ".srt")
    if guess.exists() and not overwrite:
        print(f"[OCR] SRT exists, skip extract: {guess}")
        return guess

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "third_party" / "video-subtitle-extractor" / "scripts" / "extract_srt.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Missing subtitle extractor script: {script}\n"
            "Expected vendor path: third_party/video-subtitle-extractor (copy from your OCR toolchain)."
        )

    cmd = [
        sys.executable,
        str(script),
        "--video",
        video_path,
        "--language",
        language,
        "--mode",
        mode,
        "--output-dir",
        str(out_dir),
    ]
    if full_image:
        cmd += ["--filter-by-area", "false", "--split-regions", "false"]
    stdout = run_cmd(cmd)
    path = Path(stdout.splitlines()[-1].strip())
    if path.exists():
        return path
    if guess.exists():
        return guess
    raise FileNotFoundError(f"SRT not found from extractor output: {stdout}")


def _strip_cjk_from_srt(srt_path: Path) -> None:
    import re

    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out_lines: list[str] = []
    cjk = re.compile(r"[\u4E00-\u9FFF]+")
    for ln in lines:
        if "-->" not in ln and not ln.strip().isdigit():
            ln = cjk.sub("", ln)
        out_lines.append(ln)
    srt_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline orchestrator: OCR + Clip -> unified semantic index, then optional summary")
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--embed-model", default="text-embedding-3-large")
    ap.add_argument("--clip-len-sec", type=float, default=10.0)
    ap.add_argument("--clip-sample-fps", type=float, default=2.0)
    ap.add_argument("--clip-max-frames", type=int, default=None, help="Max frames per clip window (defaults to 32 if unset)")
    ap.add_argument("--clip-workers", type=int, default=4, help="Caption threads for clip builder")
    ap.add_argument("--indexes-root", default=None, help="Root for indexes (default: env INDEXES_ROOT or <repo>/data/indexes)")
    ap.add_argument("--summaries-root", default=None, help="Root for summaries (default: env SUMMARIES_ROOT or <repo>/data/summaries)")
    ap.add_argument("--cache-dir", default=None, help="Cache root dir (default: env CACHE_DIR or <repo>/data/cache)")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-summary", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite-ocr", action="store_true")
    ap.add_argument("--overwrite-clip", action="store_true")
    ap.add_argument("--overwrite-summary", action="store_true")
    ap.add_argument("--ocr-full-image", action="store_true", help="Use full-image OCR (no region split/filter)")
    ap.add_argument("--ocr-language", default=None, help="OCR language code (e.g., en/ch/japan/korean). Default 'ch'.")
    ap.add_argument("--ocr-exclude-chinese", action="store_true", help="Remove CJK characters from SRT text lines after extraction")
    ap.add_argument("--existing-srt-only", action="store_true", help="Use cached SRT if present; skip OCR stage when missing")
    args = ap.parse_args()

    video_path = Path(args.video).as_posix()
    video_id = args.video_id or _slugify(Path(video_path).stem)

    indexes_root = Path(args.indexes_root).expanduser().resolve() if args.indexes_root else get_indexes_root().resolve()
    summaries_root = Path(args.summaries_root).expanduser().resolve() if args.summaries_root else get_summaries_root().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else get_cache_root().resolve()

    temp_root = cache_root / "semantic_inputs"
    temp_ocr_root = temp_root / "ocr"
    temp_clip_root = temp_root / "clip"
    _ensure_dir(temp_ocr_root)
    _ensure_dir(temp_clip_root)

    ow_ocr = bool(args.overwrite or args.overwrite_ocr)
    ow_clip = bool(args.overwrite or args.overwrite_clip)
    ow_summary = bool(args.overwrite or args.overwrite_summary)

    # 1) OCR -> cache/semantic_inputs/ocr/<video_id>
    if not args.skip_ocr:
        ocr_dir = temp_ocr_root / video_id
        ocr_caps = ocr_dir / "ocr_captions.json"
        ocr_vec = ocr_dir / "ocr_vectors.npy"
        ocr_norms = ocr_dir / "ocr_norms.npy"
        ocr_ids = ocr_dir / "ocr_doc_ids.txt"
        ocr_meta = ocr_dir / "ocr_meta.json"

        def _ocr_index_complete() -> bool:
            return _index_complete(ocr_caps, ocr_vec, ocr_norms, ocr_ids, ocr_meta)

        if _ocr_index_complete() and not ow_ocr:
            print(f"[OCR] skip: npy index exists -> {ocr_dir}")
        else:
            srt_dir = cache_root / "srt" / video_id
            lang = args.ocr_language or "ch"
            srt_guess = srt_dir / (Path(video_path).stem + ".srt")
            if args.existing_srt_only:
                if srt_guess.exists():
                    print(f"[OCR] existing-srt-only: using {srt_guess}")
                    srt_path: Optional[Path] = srt_guess
                else:
                    print(f"[OCR] existing-srt-only: skip, missing {srt_guess}")
                    srt_path = None
            else:
                srt_path = extract_srt(
                    video_path,
                    srt_dir,
                    language=lang,
                    overwrite=bool(ow_ocr),
                    full_image=bool(args.ocr_full_image),
                )

            if srt_path is None:
                print("[OCR] skipped build because no SRT available")
            else:
                if args.ocr_exclude_chinese:
                    _strip_cjk_from_srt(srt_path)
                    print("[OCR] Removed CJK characters from SRT (ocr-exclude-chinese)")
                res = build_ocr_index(
                    video_id=video_id,
                    srt_path=srt_path.as_posix(),
                    embedding_model=args.embed_model,
                    out_dir=temp_ocr_root.as_posix(),
                    overwrite=bool(ow_ocr),
                )
                print("[OCR]", json.dumps(res, ensure_ascii=False))

    # 2) Clip captions + index -> cache/semantic_inputs/clip/<video_id>
    if not args.skip_clip:
        max_frames = int(args.clip_max_frames) if (args.clip_max_frames and args.clip_max_frames > 0) else 32
        res = build_clip_index(
            video_path=video_path,
            video_id=video_id,
            clip_len_sec=float(args.clip_len_sec),
            sample_fps=float(args.clip_sample_fps),
            max_frames=max_frames,
            workers=int(args.clip_workers),
            embedding_model=args.embed_model,
            out_dir=temp_clip_root.as_posix(),
            overwrite=bool(ow_clip),
        )
        print("[CLIP]", json.dumps(res, ensure_ascii=False))

    # 3) Semantic index -> indexes/semantic/<video_id>
    semantic_root = indexes_root / "semantic"
    semantic_overwrite = bool(args.overwrite or args.overwrite_clip or args.overwrite_ocr)
    if args.skip_clip and args.skip_ocr:
        print("[SEMANTIC] Skipped semantic build because both clip and ocr stages are skipped.")
    else:
        _ensure_dir(semantic_root)
        sem_dir = semantic_root / video_id
        sem_caps = sem_dir / "semantic_captions.json"
        sem_vecs = sem_dir / "semantic_vectors.npy"
        if (not semantic_overwrite) and sem_caps.exists() and sem_vecs.exists():
            print(f"[SEMANTIC] Skip rebuild: captions+vectors exist -> {sem_dir}")
        else:
            build_semantic_indices(
                [video_id],
                clip_root=temp_clip_root,
                ocr_root=temp_ocr_root,
                semantic_root=semantic_root,
                model=args.embed_model,
                overwrite=bool(semantic_overwrite),
            )
            if sem_caps.exists():
                print("[SEMANTIC] Built unified semantic index:", sem_dir.as_posix())
            else:
                raise RuntimeError(f"Semantic build finished but {sem_caps} is missing (check clip/OCR inputs).")

    # 4) Global summary -> summaries/<video_id>/full_story.txt
    if not args.skip_summary:
        out_txt = build_global_summary(
            video_id=video_id,
            indexes_root=indexes_root,
            summaries_root=summaries_root,
            cache_root=cache_root,
            overwrite=bool(ow_summary),
        )
        print("[SUMMARY]", out_txt.as_posix())

    print("[DONE] video_id:", video_id)
    print("indexes root:", indexes_root)
    print("summaries root:", summaries_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
