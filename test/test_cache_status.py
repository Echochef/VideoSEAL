from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from videoseal.utils.data.check_cache_status import check_semantic, check_summary
from videoseal.utils.data.cleanup_clip_cache import collect_invalid


class TestCacheStatus(unittest.TestCase):
    def test_check_summary_missing_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vid = Path("Bench/vid1")
            out = check_summary(root, vid)
            self.assertEqual(out["summary_missing"], 1)
            self.assertEqual(out["summary_empty"], 1)

            story = root / vid / "full_story.txt"
            story.parent.mkdir(parents=True, exist_ok=True)
            story.write_text("   \n", encoding="utf-8")
            out2 = check_summary(root, vid)
            self.assertEqual(out2["summary_missing"], 0)
            self.assertEqual(out2["summary_empty"], 1)

            story.write_text("hello\n", encoding="utf-8")
            out3 = check_summary(root, vid)
            self.assertEqual(out3["summary_missing"], 0)
            self.assertEqual(out3["summary_empty"], 0)

    def test_check_semantic_error_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_root = root / "cache"
            semantic_root = root / "indexes" / "semantic"
            vid = Path("Bench/vid2")

            # semantic captions contain transient error token => final_error=1
            sem_dir = semantic_root / vid
            sem_dir.mkdir(parents=True, exist_ok=True)
            (sem_dir / "semantic_captions.json").write_text(
                '{"0_10":{"caption":"[ERROR] HTTPError: 429"},"subject_registry":{}}',
                encoding="utf-8",
            )

            # clip ckpt contains error token => ckpt_error=1
            ckpt_dir = cache_root / "captions_ckpt" / vid
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "0_10.json").write_text('{"clip_description":"Forbidden"}', encoding="utf-8")

            out = check_semantic(cache_root, semantic_root, vid)
            self.assertEqual(out["clip_final_missing"], 0)
            self.assertEqual(out["clip_final_error"], 1)
            self.assertEqual(out["clip_ckpt_error"], 1)

    def test_no_frames_sampled_is_flagged_in_semantic_caps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_root = root / "cache"
            semantic_root = root / "indexes" / "semantic"
            vid = Path("Bench/vid3")

            sem_dir = semantic_root / vid
            sem_dir.mkdir(parents=True, exist_ok=True)
            (sem_dir / "semantic_captions.json").write_text(
                '{"0_10":{"caption":"[ERROR] No frames sampled"},"subject_registry":{}}',
                encoding="utf-8",
            )

            out = check_semantic(cache_root, semantic_root, vid)
            self.assertEqual(out["clip_final_missing"], 0)
            self.assertEqual(out["clip_final_error"], 1)

    def test_collect_invalid_ckpts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt_dir = root / "captions_ckpt" / "Bench" / "vid4"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "ok.json").write_text('{"clip_description":"normal"}', encoding="utf-8")
            (ckpt_dir / "bad.json").write_text('{"clip_description":"HTTPError: 500"}', encoding="utf-8")
            (ckpt_dir / "empty.json").write_text("   ", encoding="utf-8")

            bad = collect_invalid(ckpt_dir)
            names = sorted([p.name for p in bad])
            self.assertEqual(names, ["bad.json", "empty.json"])


if __name__ == "__main__":
    unittest.main()
