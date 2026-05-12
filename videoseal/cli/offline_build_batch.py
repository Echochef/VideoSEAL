from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from videoseal.utils.data.check_cache_status import check_semantic, check_summary
from videoseal.utils.data.cleanup_clip_cache import collect_invalid
from videoseal.utils.env_paths import get_cache_root, get_frames_root, get_indexes_root, get_summaries_root


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _slugify(stem: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "-", (stem or "").strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-").lower()


def _resolve_video_id(*, video_path: Path, benchmark: str, explicit_video_id: str) -> str:
    vid = (explicit_video_id or "").strip() or _slugify(video_path.stem)
    bench = (benchmark or "").strip()
    if bench and not vid.startswith(f"{bench}/"):
        vid = f"{bench}/{vid}"
    return vid


def _maybe_set_ark_upload_prefix(env: dict[str, str], *, video_id: str) -> None:
    if (env.get("MLLM_BACKEND") or "").strip() != "ark":
        return
    base_prefix = (env.get("MLLM_UPLOAD_PREFIX_BASE") or "").strip().rstrip("/")
    env["MLLM_UPLOAD_PREFIX"] = f"{base_prefix}/{video_id}" if base_prefix else video_id


def _cleanup_invalid_clip_ckpts(cache_root: Path, *, video_id: str) -> int:
    ckpt_dir = cache_root / "captions_ckpt" / Path(video_id)
    bad = collect_invalid(ckpt_dir)
    removed = 0
    for p in bad:
        p.unlink(missing_ok=True)
        removed += 1
    return removed


def _remove_invalid_semantic_captions(indexes_root: Path, *, video_id: str) -> None:
    sem_caps = indexes_root / "semantic" / Path(video_id) / "semantic_captions.json"
    if sem_caps.exists():
        sem_caps.unlink()


def _run_one_video(
    *,
    repo_root: Path,
    video_path: Path,
    benchmark: str,
    explicit_video_id: str,
    args: argparse.Namespace,
) -> None:
    env = dict(os.environ)

    indexes_root = Path(args.indexes_root).expanduser().resolve() if args.indexes_root else get_indexes_root().resolve()
    summaries_root = Path(args.summaries_root).expanduser().resolve() if args.summaries_root else get_summaries_root().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else get_cache_root().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve() if args.frames_root else get_frames_root().resolve()

    video_id = _resolve_video_id(video_path=video_path, benchmark=benchmark, explicit_video_id=explicit_video_id)
    _maybe_set_ark_upload_prefix(env, video_id=video_id)

    status_summary = check_summary(summaries_root, Path(video_id))
    status_sem = check_semantic(cache_root, indexes_root / "semantic", Path(video_id))

    ow_summary = bool(args.overwrite_summary or args.overwrite)
    ow_ocr = bool(args.overwrite_ocr or args.overwrite)
    ow_clip = bool(args.overwrite_clip or args.overwrite)

    if not ow_summary and (status_summary.get("summary_missing") == 1 or status_summary.get("summary_empty") == 1):
        ow_summary = True

    if status_sem.get("clip_ckpt_error") == 1:
        _cleanup_invalid_clip_ckpts(cache_root, video_id=video_id)

    if status_sem.get("clip_final_error") == 1:
        _remove_invalid_semantic_captions(indexes_root, video_id=video_id)

    cmd = [
        str(args.python),
        "-m",
        "videoseal.offline_build.orchestrator",
        "--video",
        video_path.as_posix(),
        "--video-id",
        video_id,
        "--embed-model",
        str(args.embed_model),
        "--clip-len-sec",
        str(args.clip_len_sec),
        "--clip-sample-fps",
        str(args.clip_sample_fps),
        "--clip-workers",
        str(args.clip_workers),
        "--indexes-root",
        indexes_root.as_posix(),
        "--summaries-root",
        summaries_root.as_posix(),
        "--cache-dir",
        cache_root.as_posix(),
    ]
    if args.clip_max_frames is not None:
        cmd += ["--clip-max-frames", str(args.clip_max_frames)]
    if args.skip_ocr:
        cmd += ["--skip-ocr"]
    if args.skip_clip:
        cmd += ["--skip-clip"]
    if args.skip_summary:
        cmd += ["--skip-summary"]
    if args.ocr_full_image:
        cmd += ["--ocr-full-image"]
    if args.existing_srt_only:
        cmd += ["--existing-srt-only"]
    if args.ocr_exclude_chinese:
        cmd += ["--ocr-exclude-chinese"]
    if ow_ocr:
        cmd += ["--overwrite-ocr"]
    if ow_clip:
        cmd += ["--overwrite-clip"]
    if ow_summary:
        cmd += ["--overwrite-summary"]

    subprocess.run(cmd, cwd=repo_root.as_posix(), env=env, check=True)

    if args.clean_frames:
        frames_dir = frames_root / Path(video_id)
        if frames_dir.exists() and frames_root in frames_dir.parents:
            shutil.rmtree(frames_dir, ignore_errors=False)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description="Offline build batch runner (OCR + clip -> semantic index + optional summary).")
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"))
    ap.add_argument("--video", default=os.getenv("VIDEO", ""), help="Video file path or a directory of videos.")
    ap.add_argument("--benchmark", default=os.getenv("BENCHMARK", ""))
    ap.add_argument("--video-id", default=os.getenv("VIDEO_ID", ""))
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")))
    ap.add_argument("--clip-len-sec", type=float, default=float(os.getenv("CLIP_LEN_SEC", "16")))
    ap.add_argument("--clip-sample-fps", type=float, default=float(os.getenv("CLIP_SAMPLE_FPS", "1")))
    ap.add_argument("--clip-workers", type=int, default=int(os.getenv("CLIP_WORKERS", "32")))
    ap.add_argument("--clip-max-frames", type=int, default=int(os.getenv("CLIP_MAX_FRAMES")) if os.getenv("CLIP_MAX_FRAMES") else None)
    ap.add_argument("--video-workers", type=int, default=int(os.getenv("VIDEO_WORKERS", "16")))

    ap.add_argument("--indexes-root", default=os.getenv("INDEXES_ROOT"))
    ap.add_argument("--summaries-root", default=os.getenv("SUMMARIES_ROOT"))
    ap.add_argument("--cache-dir", default=os.getenv("CACHE_DIR"))
    ap.add_argument("--frames-root", default=os.getenv("FRAMES_ROOT"))

    ap.add_argument("--overwrite", action="store_true", default=(os.getenv("OVERWRITE", "0") == "1"))
    ap.add_argument("--overwrite-ocr", action="store_true", default=(os.getenv("OVERWRITE_OCR", "0") == "1"))
    ap.add_argument("--overwrite-clip", action="store_true", default=(os.getenv("OVERWRITE_CLIP", "0") == "1"))
    ap.add_argument("--overwrite-summary", action="store_true", default=(os.getenv("OVERWRITE_SUMMARY", "0") == "1"))
    ap.add_argument("--skip-ocr", action="store_true", default=(os.getenv("SKIP_OCR", "0") == "1"))
    ap.add_argument("--skip-clip", action="store_true", default=(os.getenv("SKIP_CLIP", "0") == "1"))
    ap.add_argument("--skip-summary", action="store_true", default=(os.getenv("SKIP_SUMMARY", "1") == "1"))
    ap.add_argument("--existing-srt-only", action="store_true", default=(os.getenv("USE_EXISTING_SRT_ONLY", "1") == "1"))
    ap.add_argument("--ocr-full-image", action="store_true", default=(os.getenv("OCR_FULL_IMAGE", "0") == "1"))
    ap.add_argument("--ocr-exclude-chinese", action="store_true", default=(os.getenv("OCR_EXCLUDE_CHINESE", "0") == "1"))
    ap.add_argument("--clean-frames", action="store_true", default=(os.getenv("CLEAN_FRAMES", "0") == "1"))
    args = ap.parse_args()

    if not args.video:
        raise SystemExit("Missing --video (or env VIDEO).")

    video = Path(args.video).expanduser()
    if video.is_dir():
        files = sorted([p for p in video.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        if not files:
            raise SystemExit(f"No videos found under: {video}")
        workers = max(1, int(args.video_workers))
        if workers == 1:
            for p in files:
                _run_one_video(repo_root=repo_root, video_path=p, benchmark=args.benchmark, explicit_video_id=args.video_id, args=args)
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for p in files:
                    futures.append(
                        ex.submit(
                            _run_one_video,
                            repo_root=repo_root,
                            video_path=p,
                            benchmark=args.benchmark,
                            explicit_video_id=args.video_id,
                            args=args,
                        )
                    )
                for fut in as_completed(futures):
                    fut.result()
    else:
        _run_one_video(repo_root=repo_root, video_path=video, benchmark=args.benchmark, explicit_video_id=args.video_id, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
