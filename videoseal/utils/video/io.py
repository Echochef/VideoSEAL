from __future__ import annotations

import os

import cv2


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using OpenCV metadata.

    Returns 0.0 if read fails.
    """
    env_dur = os.getenv("VIDEO_DURATION_SEC")
    if env_dur is not None and str(env_dur).strip():
        try:
            v = float(str(env_dur).strip())
            if v > 0.0:
                return v
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0:
        return 0.0
    return float(frame_count / fps)

