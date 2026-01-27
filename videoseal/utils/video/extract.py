from __future__ import annotations

import os
from typing import Optional


def extract_clip(video_path: str, start_sec: float, end_sec: float, out_path: Optional[str] = None) -> str:
    """Extract [start_sec, end_sec] to a new mp4 using moviepy."""
    from moviepy.video.io.VideoFileClip import VideoFileClip

    if out_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(os.path.dirname(video_path), f"{base}_{start_sec:.2f}_{end_sec:.2f}.mp4")

    with VideoFileClip(video_path) as clip:
        dur = clip.duration or 0.0
        s = max(0.0, min(start_sec, dur))
        e = max(0.0, min(end_sec, dur))
        if e <= s:
            e = min(s + 1.0, dur)
        sub = clip.subclip(s, e)
        sub.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
    return out_path

