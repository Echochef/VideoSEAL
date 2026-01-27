import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


class _StubVLM:
    def generate_images_paths(self, image_paths, prompt, **kwargs):
        assert image_paths
        assert isinstance(prompt, str) and prompt.strip()
        return "Answer: A\nConfidence: 0.99"


class TestVisualInspect(unittest.TestCase):
    def test_visual_inspect_smoke(self) -> None:
        from videoseal.tools import visual_tools as vt
        from videoseal.tools.visual_tools import VisualInspectAliasTool

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "toy.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
            for i in range(30):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = i * 3
                frame[:, :, 1] = 255 - i * 3
                out.write(frame)
            out.release()

            env_old = dict(os.environ)
            try:
                os.environ["CLEAN_FRAMES"] = "1"
                os.environ["CLEAN_FRAMES_ASYNC"] = "0"
                os.environ["INSPECT_FPS"] = "2"
                os.environ["INSPECT_MAX_TOTAL_IMAGES"] = "8"
                os.environ["VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE"] = "0"

                vt.FRAMES_ROOT = tmp / "frames"
                with patch("videoseal.tools.visual_tools.build_mllm_client_from_env_prefix", return_value=_StubVLM()):
                    res = VisualInspectAliasTool().forward(
                        video_path=str(video_path),
                        spans=[{"start_time": "00:00:00", "end_time": "00:00:01"}],
                        questions="What is shown? (A) foo (B) bar (C) baz (D) qux",
                    )
                self.assertIsNone(res.error)
                self.assertIn("answer", res.output)
                self.assertIn("spans", res.output)
                self.assertTrue(str(res.output["answer"]).startswith("Answer:"))
                self.assertEqual(len(res.output["spans"]), 1)
            finally:
                os.environ.clear()
                os.environ.update(env_old)

