from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

from ..api.mllm import MLLMClient, _b64_of_image_bytes
from videoseal.prompts.caption_prompts import default_caption_prompt
from ..video import sample_uniform_frames
from ..video.time import sec_to_hhmmss


# --- Lightweight Tool & ToolOutput (rllm-style), kept local for decoupling ---
@dataclass
class ToolOutput:
    name: str
    output: Any | None = None
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if self.output is None:
            return ""
        if isinstance(self.output, (list, dict)):
            import json

            return json.dumps(self.output)
        return str(self.output)

    def to_string(self) -> str:
        return str(self)


class Tool:
    """Minimal tool base compatible with rllm-style tool calling."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    def json(self) -> dict[str, Any]:  # to be overridden
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def forward(self, *args, **kwargs) -> ToolOutput:  # to be overridden
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> ToolOutput:
        return self.forward(*args, **kwargs)


# use shared time utility


def _sample_clip_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps_sample: float = 2.0,
    max_frames: int = 32,
) -> List[bytes]:
    """
    Uniformly sample frames in [start_sec, end_sec) and return list of JPEG bytes.
    Delegates to common sampler and encodes images.
    """
    import cv2

    paths, _ = sample_uniform_frames(
        video_path,
        float(start_sec),
        float(end_sec),
        fps=float(fps_sample),
        max_frames=int(max_frames),
        output_dir=None,
    )
    imgs: List[bytes] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            imgs.append(buf.tobytes())
        if len(imgs) >= max_frames:
            break
    if not imgs:
        raise ValueError("No frames sampled from the given interval")
    return imgs


class VideoClipCaptionTool(Tool):
    """Visual-only clip captioning tool.

    Reads frames from a given video interval (no subtitles/audio) and calls a multimodal
    LLM to produce entities and a clip-level visual description. The model is asked to
    return a JSON string with keys: clip_start_time, clip_end_time, entities[], clip_description.
    """

    def __init__(self, name: str = "video_clip_caption", description: str | None = None, client: Optional[MLLMClient] = None):
        super().__init__(
            name=name,
            description=description
            or "Generate visual-only entity and clip description for a video interval (no transcript).",
        )
        self.client = client or MLLMClient()

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Absolute path to the video file."},
                        "start_sec": {"type": "number", "description": "Start time in seconds (inclusive)."},
                        "end_sec": {"type": "number", "description": "End time in seconds (exclusive)."},
                        "fps_sample": {"type": "number", "description": "Sampling FPS for frame extraction (default 2.0)."},
                        "max_frames": {"type": "integer", "description": "Max frames to send (default 32)."},
                    },
                    "required": ["video_path", "start_sec", "end_sec"],
                },
            },
        }

    def forward(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        fps_sample: float = 2.0,
        max_frames: int = 32,
    ) -> ToolOutput:
        try:
            import os
            prompt = default_caption_prompt(sec_to_hhmmss(start_sec), sec_to_hhmmss(end_sec))
            backend = os.getenv("MLLM_BACKEND", "openai").lower()
            if backend in ("dashscope", "qwen", "qwen_mm"):
                # dashscope: pass frame file paths directly to the client
                paths, _ = sample_uniform_frames(
                    video_path,
                    float(start_sec),
                    float(end_sec),
                    fps=float(fps_sample),
                    max_frames=int(max_frames),
                    output_dir=None,
                )
                text = self.client.generate_images_paths(paths, prompt, response_json=True)
            else:
                imgs = _sample_clip_frames(video_path, start_sec, end_sec, fps_sample=fps_sample, max_frames=max_frames)
                b64_list = [_b64_of_image_bytes(b) for b in imgs]
                text = self.client.generate_caption(b64_list, prompt, response_json=True)
            return ToolOutput(name=self.name, output=text)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")


# Convenience factory for external registries
def build_tools() -> dict[str, type[VideoClipCaptionTool]]:
    """Return a mapping suitable for a tool registry: {"video_clip_caption": VideoClipCaptionTool}.

    Example usage with a registry or MultiTool-like aggregator:
        tool_map = build_tools()
        # registry.register_all(tool_map) or MultiTool(tool_map=tool_map)
    """
    return {"video_clip_caption": VideoClipCaptionTool}
