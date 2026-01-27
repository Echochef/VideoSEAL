from __future__ import annotations

import argparse
import json
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
        meta_json = json.loads(meta_json)
    if meta_json is None:
        return None
    if not isinstance(meta_json, dict):
        raise TypeError(f"extra_info.meta_json must be a dict or a JSON string encoding a dict; got {type(meta_json).__name__}.")
    v = _norm(meta_json.get("benchmark"))
    return v or None


def _optional_positive_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    x = float(v)  # type: ignore[arg-type]
    return x if x > 0.0 else None


def load_tasks_from_parquet(
    parquet_path: Path,
    *,
    video_id: Optional[str] = None,
    uids: Optional[set[str]] = None,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> List[Dict[str, Any]]:
    t = pq.read_table(parquet_path)
    cols = {name: t[name].to_pylist() for name in t.column_names}
    n = t.num_rows

    vid_filter = _norm(video_id) if video_id else None
    tasks: List[Dict[str, Any]] = []

    for i in range(n):
        if shard_index is not None and num_shards is not None and num_shards > 1:
            if i % num_shards != shard_index:
                continue
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

        vdur = _optional_positive_float(info.get("video_duration_sec"))
        cdur = _optional_positive_float(info.get("clip_duration_sec"))
        tref = _norm(info.get("time_reference"))

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
                "clip_duration_sec": cdur,
                "time_reference": tref or None,
                "visual_index": vis or None,
                "summary_dir": sdir or None,
                "summary_file": sfile or None,
                "subtitle_path": subpath or None,
                "benchmark": bench or None,
            }
        )
    return tasks


def _has_pred_result(task: Dict[str, Any], *, save_root: Path) -> bool:
    vid = str(task["video_id"])
    uid = str(task["uid"])
    pred_path = save_root / vid / "preds" / f"{uid}.json"
    if not pred_path.exists():
        return False
    rec = json.loads(pred_path.read_text(encoding="utf-8"))
    pred = str(rec.get("pred") or "").strip()
    return bool(pred)


def _run_one_task(task: Dict[str, Any], *, save_root: Path, max_steps: int, prompt_type: int = 0) -> Tuple[str, str, str]:
    from videoseal.tools.tool_map import build_tool_map
    from videoseal.agents.tool_agent import SimpleToolAgent

    vid = str(task["video_id"])
    uid = str(task["uid"])
    q = str(task["question"])
    vpath = str(task["video_path"])
    vis = task.get("visual_index") or None
    gt = str(task.get("gt") or "")
    tref = _norm(task.get("time_reference")) if task.get("time_reference") is not None else ""

    sdir = task.get("summary_dir") or None
    sfile = task.get("summary_file") or None
    subpath = task.get("subtitle_path") or None
    bench = task.get("benchmark") or None

    for k in ("SUMMARY_FILE", "SUMMARY_DIR", "SUBTITLE_PATH", "BENCHMARK", "VIDEO_DURATION_SEC", "SEMANTIC_INDEX", "VISUAL_INDEX_DIR"):
        os.environ.pop(k, None)
    vdur = task.get("video_duration_sec")
    if vdur is not None:
        os.environ["VIDEO_DURATION_SEC"] = str(float(vdur))
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

    T = build_tool_map()
    online = {k: v for k, v in T.items() if k in ("visual_retrieve", "visual_inspect")}
    agent = SimpleToolAgent(tools=online)

    res = agent.run(
        question=q,
        uid=uid,
        time_reference=tref or None,
        video_id=vid,
        video_path=vpath,
        visual_index=vis,
        groundtruth=gt,
        max_steps=max_steps,
        save_dir=str(out_dir),
        prompt_type=prompt_type,
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
        json.dump(
            {"uid": uid, "video_id": vid, "question": q, "pred": pred, "gt": gt, "time_reference": tref},
            f,
            ensure_ascii=False,
        )
    return vid, uid, pred


def _worker_one_task(*, task: Dict[str, Any], save_root: str, max_steps: int, prompt_type: int, result_queue) -> None:
    import os as _os

    pid = int(_os.getpid())
    vid = str(task.get("video_id") or "")
    uid = str(task.get("uid") or "")
    try:
        v, u, pred = _run_one_task(task, save_root=Path(save_root), max_steps=int(max_steps), prompt_type=int(prompt_type))
        result_queue.put({"pid": pid, "video_id": v, "uid": u, "pred": pred})
    except Exception as e:
        result_queue.put({"pid": pid, "video_id": vid, "uid": uid, "error": f"{type(e).__name__}: {e}"})


@dataclass(frozen=True)
class _RunningProc:
    pid: int
    proc: Any
    task: Dict[str, Any]
    start_time: float


def aggregate_answers(runs_root: Path, vids: List[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    g_total_answered = 0
    g_correct = 0
    g_count = 0
    uniq_vids = sorted({v for v in vids if v})
    for vid in uniq_vids:
        vdir = runs_root / vid
        preds_dir = vdir / "preds"
        answers: Dict[str, str] = {}
        pairs: List[Dict[str, str]] = []
        total = 0
        correct = 0
        if preds_dir.exists():
            for p in preds_dir.glob("*.json"):
                rec = json.loads(p.read_text(encoding="utf-8"))
                uid = str(rec.get("uid") or "")
                pred = str(rec.get("pred") or "")
                gt = str(rec.get("gt") or "")
                tref = str(rec.get("time_reference") or "")
                if not uid:
                    continue
                answers[uid] = pred
                if pred:
                    total += 1
                    if gt and pred.upper() == gt.upper():
                        correct += 1
                pair_rec: Dict[str, str] = {"uid": uid, "pred": pred, "gt": gt}
                if tref:
                    pair_rec["time_reference"] = tref
                pairs.append(pair_rec)
        acc = (correct / total) if total > 0 else 0.0
        out_obj = {"video_id": vid, "count": len(answers), "answers": answers, "stats": {"total_answered": total, "correct": correct, "acc": acc}, "pairs": pairs}
        (vdir / "answers.json").write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"video_id": vid, "answers": (vdir / "answers.json").as_posix(), "acc": acc, "count": len(answers)})
        g_total_answered += total
        g_correct += correct
        g_count += len(answers)
    g_acc = (g_correct / g_total_answered) if g_total_answered > 0 else 0.0
    global_summary = {"videos": len(uniq_vids), "count": g_count, "total_answered": g_total_answered, "correct": g_correct, "acc": g_acc}
    (runs_root / "_global_summary.json").write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, global_summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-question concurrent runner from parquet (API backend)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--uids-file", default=None)
    ap.add_argument("--save-runs", default=None)
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "8")))
    ap.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "8")))
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--cn-prompt", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--skip-existing", action="store_true")
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

    shard_index: Optional[int] = args.shard_index
    num_shards: Optional[int] = args.num_shards
    if (shard_index is None) ^ (num_shards is None):
        raise SystemExit("--shard-index and --num-shards must be provided together.")
    if num_shards is not None:
        if num_shards <= 0:
            raise SystemExit("--num-shards must be a positive integer.")
        if shard_index is None or not (0 <= shard_index < num_shards):
            raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards.")

    tasks_all = load_tasks_from_parquet(
        parquet_path,
        video_id=args.video_id,
        uids=uids,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if not tasks_all:
        raise SystemExit(f"No tasks found in {parquet_path} (video-id={args.video_id or '*'}).")

    skipped: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    if args.skip_existing:
        for t in tasks_all:
            if _has_pred_result(t, save_root=runs_root):
                skipped.append(t)
            else:
                tasks.append(t)
    else:
        tasks = list(tasks_all)

    print(json.dumps({"skipped": len(skipped), "remaining": len(tasks)}, ensure_ascii=False))
    if not tasks:
        raise SystemExit("All tasks already have pred results.")

    vids = [t["video_id"] for t in tasks]
    print(json.dumps({"tasks": len(tasks), "videos": sorted(list(set(vids)))}, ensure_ascii=False))

    max_workers = max(1, int(args.concurrency))
    task_timeout = float(getattr(args, "task_timeout_sec", 0.0) or 0.0)

    import multiprocessing as mp

    mp_ctx = mp.get_context("spawn")

    def _print_rec(rec: Dict[str, Any]) -> None:
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    if task_timeout > 0:
        if mp_ctx is None:
            raise SystemExit("Hard-timeout mode requires multiprocessing context (spawn) but it is not available.")
        result_q = mp_ctx.Queue()
        running: Dict[int, _RunningProc] = {}
        done: set[tuple[str, str]] = set()
        idx = 0
        total = len(tasks)

        def _finish_once(rec: Dict[str, Any]) -> None:
            key = (str(rec.get("video_id") or ""), str(rec.get("uid") or ""))
            if key in done:
                return
            done.add(key)
            _print_rec(rec)

        try:
            while len(done) < total:
                while idx < total and len(running) < max_workers:
                    t = tasks[idx]
                    idx += 1
                    proc = mp_ctx.Process(
                        target=_worker_one_task,
                        kwargs={
                            "task": t,
                            "save_root": runs_root.as_posix(),
                            "max_steps": int(args.max_steps),
                            "prompt_type": int(args.cn_prompt),
                            "result_queue": result_q,
                        },
                    )
                    proc.daemon = True
                    proc.start()
                    running[int(proc.pid)] = _RunningProc(pid=int(proc.pid), proc=proc, task=t, start_time=time.time())

                try:
                    msg = result_q.get(timeout=0.2)
                except queue.Empty:
                    msg = None
                if isinstance(msg, dict):
                    pid = int(msg.get("pid") or -1)
                    rp = running.pop(pid, None)
                    if rp is not None:
                        rp.proc.join(timeout=0.0)
                    msg.pop("pid", None)
                    _finish_once(msg)

                for pid, rp in list(running.items()):
                    alive = rp.proc.is_alive()
                    if not alive:
                        running.pop(pid, None)
                        rec = {"video_id": str(rp.task.get("video_id") or ""), "uid": str(rp.task.get("uid") or ""), "error": "worker exited without result"}
                        _finish_once(rec)
                        continue
                    if (time.time() - rp.start_time) > task_timeout:
                        if rp.proc.is_alive():
                            rp.proc.terminate()
                        rp.proc.join(timeout=1.0)
                        running.pop(pid, None)
                        rec = {"video_id": str(rp.task.get("video_id") or ""), "uid": str(rp.task.get("uid") or ""), "error": f"timeout>{task_timeout}s"}
                        _finish_once(rec)
        finally:
            for pid, rp in list(running.items()):
                if rp.proc.is_alive():
                    rp.proc.terminate()
                rp.proc.join(timeout=0.2)
                running.pop(pid, None)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
            futs = [ex.submit(_run_one_task, t, save_root=runs_root, max_steps=int(args.max_steps), prompt_type=int(args.cn_prompt)) for t in tasks]
            for fut in as_completed(futs):
                try:
                    vid, uid, pred = fut.result()
                    _print_rec({"video_id": vid, "uid": uid, "pred": pred})
                except Exception as e:
                    _print_rec({"error": f"{type(e).__name__}: {e}"})

    if not args.no_aggregate:
        _summaries, global_summary = aggregate_answers(runs_root, vids)
        print(json.dumps({"global": global_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
