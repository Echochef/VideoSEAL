from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from videoseal.offline_build.semantic_index_builder import build_semantic_indices


class TestSemanticIndexBuilder(unittest.TestCase):
    def test_merge_clip_and_ocr_into_semantic_captions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip_root = root / "clip"
            ocr_root = root / "ocr"
            semantic_root = root / "semantic"
            vid = "Bench/vid1"

            (clip_root / vid).mkdir(parents=True, exist_ok=True)
            (ocr_root / vid).mkdir(parents=True, exist_ok=True)

            (clip_root / vid / "captions.json").write_text(
                json.dumps(
                    {
                        "0_10": {"caption": "A person enters the room.", "entities": []},
                        "10_20": {"caption": "They sit down.", "entities": []},
                        "subject_registry": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (ocr_root / vid / "ocr_captions.json").write_text(
                json.dumps(
                    {
                        "5_6": {"caption": "Hello", "entities": []},
                        "15_16": {"caption": "Bye", "entities": []},
                        "subject_registry": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            old = os.environ.get("SKIP_SEMANTIC_EMBED")
            os.environ["SKIP_SEMANTIC_EMBED"] = "1"
            self.addCleanup(lambda: os.environ.__setitem__("SKIP_SEMANTIC_EMBED", old) if old is not None else os.environ.pop("SKIP_SEMANTIC_EMBED", None))
            build_semantic_indices(
                [vid],
                clip_root=clip_root,
                ocr_root=ocr_root,
                semantic_root=semantic_root,
                model="text-embedding-3-large",
                overwrite=True,
            )

            out_caps = semantic_root / vid / "semantic_captions.json"
            out_txt = semantic_root / vid / "semantic_segments.txt"
            self.assertTrue(out_caps.exists())
            self.assertTrue(out_txt.exists())

            data = json.loads(out_caps.read_text(encoding="utf-8"))
            self.assertIn("0_10", data)
            self.assertIn("Visual:", data["0_10"]["caption"])
            self.assertIn("Dialogue:", data["0_10"]["caption"])

            # In SKIP_SEMANTIC_EMBED mode we only guarantee captions/text output.
            self.assertFalse((semantic_root / vid / "semantic_vectors.npy").exists())


if __name__ == "__main__":
    unittest.main()
