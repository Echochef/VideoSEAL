from __future__ import annotations

from typing import List

# Reuse common sampler; convert saved JPEG paths to base64 strings
from pathlib import Path
from ..video import sample_uniform_frames
from ..env_paths import get_frames_root


def sample_frames_b64(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps_sample: float = 2.0,
    max_frames: int = 32,
    resize_scale: float = 0.75,
) -> List[str]:
    """
    Uniformly sample frames in [start_sec, end_sec) and return list of base64 (JPEG) strings.
    Delegates actual sampling to common utils, then encodes to base64.
    """
    import base64
    import cv2

    # Use common sampler to write JPEGs and get paths
    # central frames dir under data/frames/<video_id>
    def _slugify(stem: str) -> str:
        import re
        s = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip())
        s = re.sub(r"-+", "-", s)
        return s.strip("-").lower()

    vid_id = _slugify(Path(video_path).stem)
    frames_root = get_frames_root() / vid_id
    frames_root.mkdir(parents=True, exist_ok=True)

    paths, _ = sample_uniform_frames(
        video_path,
        float(start_sec),
        float(end_sec),
        fps=float(fps_sample),
        max_frames=int(max_frames),
        output_dir=str(frames_root),
    )

    out: List[str] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        # Optional resize
        if resize_scale and resize_scale > 0:
            img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        out.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        if len(out) >= max_frames:
            break
    if not out:
        raise ValueError("No frames sampled from the given interval")
    return out
