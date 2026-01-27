from __future__ import annotations

import os
import textwrap
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .subtitles import SubtitleIndex, build_subtitle_index


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_font(size_px: int) -> ImageFont.ImageFont:
    cand = [str(os.getenv("FRAME_SUBTITLE_FONT", "")).strip()]
    cand += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in cand:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, int(size_px))
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.truetype(cand[1], int(size_px)) if len(cand) > 1 else ImageFont.load_default()


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, *, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    text = " ".join([str(x).strip() for x in str(text).splitlines() if str(x).strip()]).strip()
    if not text:
        return []

    def _measure(t: str) -> int:
        try:
            return int(draw.textlength(t, font=font))
        except Exception:
            try:
                box = draw.textbbox((0, 0), t, font=font)
                return int(box[2] - box[0])
            except Exception:
                return len(t) * max(1, int(getattr(font, "size", 12)))

    if _measure(text) <= max_width:
        return [text]

    words = text.split(" ")
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        if not w:
            continue
        trial = (" ".join(cur + [w])).strip()
        if cur and _measure(trial) > max_width:
            lines.append(" ".join(cur).strip())
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur).strip())

    if len(lines) == 1 and _measure(lines[0]) > max_width:
        approx_chars = max(int(max_width / max(6, int(getattr(font, "size", 12) * 0.6))), 8)
        lines = textwrap.wrap(text, width=approx_chars, break_long_words=True, break_on_hyphens=False)
    return [l for l in lines if l]


def _burn_subtitle(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    t = str(text or "").strip()
    if not t:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    font_size = int(round(max(14.0, min(48.0, h * 0.045))))

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _pick_font(font_size)

    pad_x = max(10, int(round(w * 0.02)))
    pad_y = max(8, int(round(h * 0.015)))
    max_width = max(20, w - 2 * pad_x)

    lines = _wrap_text_to_width(d, t, font=font, max_width=max_width)
    if not lines:
        return frame_bgr

    max_lines = int(os.getenv("FRAME_SUBTITLE_MAX_LINES", "3") or 3)
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = (lines[-1].rstrip(".") + " …").strip()

    try:
        ascent, descent = font.getmetrics()
        line_h = int(ascent + descent + max(2, font_size * 0.15))
    except Exception:
        line_h = int(font_size * 1.3)

    strip_h = pad_y * 2 + line_h * len(lines)
    strip_h = min(strip_h, int(round(h * 0.45)))
    y0 = max(0, h - strip_h)

    d.rectangle([(0, y0), (w, h)], fill=(0, 0, 0, 170))

    y = y0 + pad_y
    for line in lines:
        d.text(
            (pad_x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_h

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out_bgr


def sample_uniform_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 2.0,
    max_frames: int = 32,
    output_dir: str | None = None,
    subtitle_path: str | None = None,
) -> Tuple[List[str], List[float]]:
    """Uniformly sample frames between [start_sec, end_sec]."""
    if end_sec <= start_sec:
        end_sec = start_sec + 1e-3
    if fps <= 0:
        fps = 2.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    num = max(int((end_sec - start_sec) * fps), 1)
    times = [start_sec + i * (1.0 / fps) for i in range(num)]
    target_idxs = sorted({max(0, int(round(t * video_fps))) for t in times})
    if not target_idxs:
        cap.release()
        raise RuntimeError(
            f"OpenCV sampling produced zero target frames (video={video_path}, window={start_sec:.3f}-{end_sec:.3f}, fps={fps})"
        )
    start_idx = target_idxs[0]
    end_idx = max(start_idx, int(round(end_sec * video_fps)))

    if output_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"dense_frames/{base}")
    _ensure_dir(output_dir)

    try:
        save_max_side = int(os.getenv("FRAME_MAX_LONG_SIDE", "0"))
    except Exception:
        save_max_side = 0

    def _maybe_resize(frame: np.ndarray) -> np.ndarray:
        if not save_max_side or save_max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        m = max(h, w)
        if m <= save_max_side:
            return frame
        if w >= h:
            new_w = save_max_side
            new_h = int(round(h * (save_max_side / w)))
        else:
            new_h = save_max_side
            new_w = int(round(w * (save_max_side / h)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    subtitle_index: SubtitleIndex | None = None
    if subtitle_path:
        subtitle_index = build_subtitle_index(str(subtitle_path))

    seek_idx = max(0, start_idx - int(video_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_idx)

    preheat_reads = 0
    max_preheat = 15
    while preheat_reads < max_preheat:
        ok, _ = cap.read()
        preheat_reads += 1
        if not ok:
            continue
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur >= start_idx:
            break

    paths: List[str] = []
    ts_list: List[float] = []
    next_k = 0

    def _ts_for_idx(idx: int) -> float:
        best_t = times[0]
        best_d = abs(idx - int(round(times[0] * video_fps)))
        for t in times[1:]:
            d = abs(idx - int(round(t * video_fps)))
            if d < best_d:
                best_d = d
                best_t = t
        return best_t

    cur_fail = 0
    max_fail = 100
    cur_idx_est = max(seek_idx, 0)
    while next_k < len(target_idxs):
        if cur_idx_est > end_idx:
            break
        ok, frame = cap.read()
        if not ok:
            cur_fail += 1
            if cur_fail > max_fail:
                break
            cur_idx_est += 1
            continue
        cur_fail = 0
        cur_idx_est = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur_idx_est < start_idx:
            continue
        while next_k < len(target_idxs) and cur_idx_est >= target_idxs[next_k]:
            ts = float(_ts_for_idx(target_idxs[next_k]))
            out = _maybe_resize(frame)
            if subtitle_index is not None:
                sub_text = subtitle_index.text_at(ts)
                if sub_text:
                    out = _burn_subtitle(out, sub_text)
            fname = os.path.join(output_dir, f"frame_{ts:.3f}.jpg")
            cv2.imwrite(fname, out)
            paths.append(fname)
            ts_list.append(ts)
            next_k += 1
            if len(paths) >= int(max_frames):
                next_k = len(target_idxs)
                break

    cap.release()
    return paths, ts_list


def sample_frames_by_clip_number(
    video_path: str,
    clip_number: int,
    clip_len_sec: int = 10,
    fps: float = 2.0,
    max_frames: int = 32,
    output_dir: str | None = None,
) -> Tuple[List[str], List[float]]:
    start_sec = max(0, int(clip_number) * int(clip_len_sec))
    end_sec = start_sec + int(clip_len_sec)
    return sample_uniform_frames(
        video_path,
        start_sec,
        end_sec,
        fps=fps,
        max_frames=max_frames,
        output_dir=output_dir,
    )


def resize_images_max_long_side(image_paths: List[str], *, max_long_side: int) -> None:
    """In-place downscale images so their longer side <= max_long_side."""
    max_side = int(max_long_side)
    if max_side <= 0:
        return

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"OpenCV failed to read image: {p}")
        h, w = img.shape[:2]
        cur = max(h, w)
        if cur <= max_side:
            continue

        if w >= h:
            new_w = max_side
            new_h = max(1, int((h * max_side) // max(1, w)))
        else:
            new_h = max_side
            new_w = max(1, int((w * max_side) // max(1, h)))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(p, resized)
        if not ok:
            raise RuntimeError(f"OpenCV failed to write image: {p}")
