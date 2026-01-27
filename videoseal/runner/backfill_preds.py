from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from videoseal.utils.lvbench_io import parse_choice_letter, parse_choice_letter_smart
from videoseal.runner.per_question_runner import aggregate_answers


def backfill_video(vdir: Path) -> Tuple[int, int]:
    video_id = vdir.name
    preds_dir = vdir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    updated = 0
    total = 0
    for tdir in sorted(vdir.glob("20*/")):
        traj_path = tdir / "trajectory.json"
        if not traj_path.exists():
            continue
        total += 1
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        uid = str(traj.get("uid") or "")
        q = str(traj.get("question") or "")
        if not uid:
            continue
        pred_path = preds_dir / f"{uid}.json"
        if pred_path.exists():
            rec = json.loads(pred_path.read_text(encoding="utf-8"))
            if str(rec.get("pred") or "").strip():
                continue
        ans = str(traj.get("answer") or "")
        pred = parse_choice_letter_smart(ans, q) or parse_choice_letter(ans) or ""
        if not pred:
            steps = traj.get("steps") or []
            for st in reversed(steps):
                msg = str((st or {}).get("model_response") or "")
                if not msg:
                    continue
                p2 = parse_choice_letter_smart(msg, q) or parse_choice_letter(msg)
                if p2:
                    pred = p2
                    break
        if pred:
            gt = str(traj.get("groundtruth") or "")
            pred_path.write_text(json.dumps({"uid": uid, "pred": pred, "gt": gt}, ensure_ascii=False), encoding="utf-8")
            updated += 1
    return updated, total


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill preds from trajectories and re-aggregate answers")
    ap.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[2] / "runs"))
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if args.video_id:
        vids: List[str] = [args.video_id]
    else:
        vids = [p.name for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]

    total_updated = 0
    for vid in vids:
        vdir = runs_root / vid
        if not vdir.is_dir():
            continue
        up, _ = backfill_video(vdir)
        total_updated += up

    _summaries, global_summary = aggregate_answers(runs_root, vids)
    print(json.dumps({"updated_preds": total_updated, "global": global_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

