import unittest
from pathlib import Path


class TestOfflineBuildBatchCLI(unittest.TestCase):
    def test_resolve_video_id(self) -> None:
        from videoseal.cli.offline_build_batch import _resolve_video_id

        p = Path("/tmp/A B.mp4")
        self.assertEqual(_resolve_video_id(video_path=p, benchmark="LVBench", explicit_video_id=""), "LVBench/a-b")
        self.assertEqual(_resolve_video_id(video_path=p, benchmark="LVBench", explicit_video_id="vid1"), "LVBench/vid1")
        self.assertEqual(_resolve_video_id(video_path=p, benchmark="LVBench", explicit_video_id="LVBench/vid1"), "LVBench/vid1")

    def test_maybe_set_ark_upload_prefix(self) -> None:
        from videoseal.cli.offline_build_batch import _maybe_set_ark_upload_prefix

        env = {"MLLM_BACKEND": "ark", "MLLM_UPLOAD_PREFIX_BASE": "prefix/"}
        _maybe_set_ark_upload_prefix(env, video_id="LVBench/vid1")
        self.assertEqual(env["MLLM_UPLOAD_PREFIX"], "prefix/LVBench/vid1")


if __name__ == "__main__":
    unittest.main()

