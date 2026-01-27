from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from videoseal.utils.agent.tool_agent_parsing import strip_multiple_choice_options
from videoseal.utils.lvbench_io import parse_choice_letter, parse_choice_letter_smart


def _norm(s: Optional[str]) -> str:
    return str(s or "").strip()


def _map_step1_dir_to_unified_semantic(step1_dir: str) -> Optional[str]:
    s = _norm(step1_dir)
    if not s:
        return None
    p = Path(s)
    parts = p.parts
    if "step1" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("step1")
    if len(parts) <= idx + 2:
        return None
    benchmark = parts[idx + 1]
    video = parts[idx + 2]
    root = Path(*parts[:idx])
    return (root / "indexes" / "semantic" / benchmark / video).as_posix()


def _meta_json_benchmark(info: Dict[str, object]) -> Optional[str]:
    meta_json = info.get("meta_json")
    if isinstance(meta_json, str):
        try:
            meta_json = json.loads(meta_json)
        except Exception:
            return None
    if isinstance(meta_json, dict):
        v = _norm(meta_json.get("benchmark"))
        return v or None
    return None


def load_tasks_from_parquet(parquet_path: Path, *, video_id: Optional[str] = None, uids: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    t = pq.read_table(parquet_path)
    cols = {name: t[name].to_pylist() for name in t.column_names}
    n = t.num_rows

    vid_filter = _norm(video_id) if video_id else None
    tasks: List[Dict[str, Any]] = []

    for i in range(n):
        extra = cols.get("extra_info", [None] * n)[i]
        if extra is None:
            continue
        info: Dict[str, object] = extra if isinstance(extra, dict) else dict(extra)
        vid = _norm(info.get("video_id"))
        if vid_filter and vid != vid_filter:
            continue
        uid = _norm(info.get("qa_uid"))
        if uids is not None and uid not in uids:
            continue
        q = _norm(info.get("question"))
        gt = _norm(info.get("ground_truth"))[:1].upper()
        vpath = _norm(info.get("VIDEO_PATH"))

        vdur_raw = info.get("video_duration_sec")
        vdur: Optional[float] = None
        if vdur_raw is not None and str(vdur_raw).strip():
            try:
                v = float(vdur_raw)
                if v > 0.0:
                    vdur = v
            except Exception:
                vdur = None

        raw_sem = _norm(info.get("SEMANTIC_INDEX")) or _norm(info.get("SEMANTIC_INDEX_DIR"))
        raw_vis = _norm(info.get("VISUAL_INDEX_DIR"))
        sem = raw_vis or _map_step1_dir_to_unified_semantic(raw_sem) or raw_sem
        vis = raw_vis or sem

        sdir = _norm(info.get("SUMMARY_DIR"))
        sfile = _norm(info.get("SUMMARY_FILE"))
        subpath = _norm(info.get("SUBTITLE_PATH"))
        bench = _norm(info.get("BENCHMARK")) or (_meta_json_benchmark(info) or "")

        if not (vid and uid and q and vpath):
            continue
        tasks.append(
            {
                "video_id": vid,
                "uid": uid,
                "question": q,
                "gt": gt,
                "video_path": vpath,
                "video_duration_sec": vdur,
                "visual_index": vis or None,
                "summary_dir": sdir or None,
                "summary_file": sfile or None,
                "subtitle_path": subpath or None,
                "benchmark": bench or None,
            }
        )
    return tasks


def _run_one_task(task: Dict[str, Any], *, save_root: Path, max_steps: int, agent=None) -> Tuple[str, str, str]:
    if agent is None:
        from videoseal.tools.tool_map import build_tool_map
        from videoseal.agents.tool_agent import SimpleToolAgent

    vid = str(task["video_id"])
    uid = str(task["uid"])
    q = str(task["question"])
    vpath = str(task["video_path"])
    vis = task.get("visual_index") or None
    gt = str(task.get("gt") or "")

    sdir = task.get("summary_dir") or None
    sfile = task.get("summary_file") or None
    subpath = task.get("subtitle_path") or None
    bench = task.get("benchmark") or None

    for k in ("SUMMARY_FILE", "SUMMARY_DIR", "SUBTITLE_PATH", "BENCHMARK", "VIDEO_DURATION_SEC", "SEMANTIC_INDEX", "VISUAL_INDEX_DIR"):
        os.environ.pop(k, None)
    vdur = task.get("video_duration_sec")
    if vdur is not None:
        try:
            os.environ["VIDEO_DURATION_SEC"] = str(float(vdur))
        except Exception:
            pass
    if sfile:
        os.environ["SUMMARY_FILE"] = str(sfile)
        os.environ["SUMMARY_DIR"] = str(Path(str(sfile)).parent)
    elif sdir:
        os.environ["SUMMARY_DIR"] = str(sdir)
    if subpath:
        os.environ["SUBTITLE_PATH"] = str(subpath)
    if bench:
        os.environ["BENCHMARK"] = str(bench)
    if vis:
        os.environ["VISUAL_INDEX_DIR"] = str(vis)

    out_dir = save_root / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    if agent is None:
        T = build_tool_map()
        online = {k: v for k, v in T.items() if k in ("visual_retrieve", "visual_inspect")}
        agent = SimpleToolAgent(tools=online)

    res = agent.run(
        question=q,
        uid=uid,
        video_id=vid,
        video_path=vpath,
        visual_index=vis,
        groundtruth=gt,
        max_steps=max_steps,
        save_dir=str(out_dir),
    )
    final_text = (res.get("final") or res.get("answer") or "").strip()
    pred = ""
    if final_text:
        q_stripped = str(q or "").strip()
        has_choices = bool(q_stripped) and strip_multiple_choice_options(q_stripped) != q_stripped
        gt_letter = str(gt or "").strip()
        is_mcq = has_choices or (len(gt_letter) == 1 and gt_letter.isalpha())
        if is_mcq:
            letter = parse_choice_letter_smart(final_text, q_stripped) or parse_choice_letter(final_text) or ""
            pred = (letter or final_text).strip().upper()
        else:
            pred = final_text.strip()

    pred_path = out_dir / "preds" / f"{uid}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump({"uid": uid, "pred": pred, "gt": gt}, f, ensure_ascii=False)
    return vid, uid, pred


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-question runner (local vLLM engine, sequential)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--uids-file", default=None)
    ap.add_argument("--save-runs", default=None)
    ap.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "8")))
    ap.add_argument("--task-timeout-sec", type=float, default=float(os.getenv("TASK_TIMEOUT_SEC", "0") or "0"))
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    runs_root = Path(args.save_runs) if args.save_runs else (Path(__file__).resolve().parents[2] / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)

    uids: Optional[set[str]] = None
    if args.uids_file:
        p = Path(str(args.uids_file)).expanduser()
        if not p.is_file():
            raise SystemExit(f"--uids-file not found: {p}")
        raw = p.read_text(encoding="utf-8").splitlines()
        uids = {line.strip() for line in raw if line.strip() and not line.strip().startswith("#")}
        if not uids:
            raise SystemExit(f"--uids-file is empty: {p}")

    tasks = load_tasks_from_parquet(parquet_path, video_id=args.video_id, uids=uids)
    if not tasks:
        raise SystemExit("No tasks found.")

    from videoseal.tools.tool_map import build_tool_map
    from videoseal.agents.tool_agent import SimpleToolAgent

    T = build_tool_map()
    online = {k: v for k, v in T.items() if k in ("visual_retrieve", "visual_inspect")}
    agent = SimpleToolAgent(tools=online)

    vids = [t["video_id"] for t in tasks]
    for t in tasks:
        vid, uid, pred = _run_one_task(t, save_root=runs_root, max_steps=int(args.max_steps), agent=agent)
        print(json.dumps({"video_id": vid, "uid": uid, "pred": pred}, ensure_ascii=False), flush=True)

    # Aggregate using the same logic as API runner for convenience.
    from videoseal.runner.per_question_runner import aggregate_answers

    _summaries, global_summary = aggregate_answers(runs_root, vids)
    print(json.dumps({"global": global_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

