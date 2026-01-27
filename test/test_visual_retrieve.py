import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TestVisualRetrieve(unittest.TestCase):
    def test_visual_retrieve_embed_index(self) -> None:
        from videoseal.tools.visual_tools import VisualRetrieveAliasTool

        with tempfile.TemporaryDirectory() as td:
            idx_dir = Path(td) / "idx"
            idx_dir.mkdir(parents=True, exist_ok=True)

            caps = {
                "0_10": {"caption": "a red car"},
                "20_30": {"caption": "a blue cat"},
                "40_50": {"caption": "a green tree"},
                "subject_registry": {},
            }
            (idx_dir / "semantic_captions.json").write_text(json.dumps(caps, ensure_ascii=False), encoding="utf-8")

            doc_ids = ["0_10", "20_30", "40_50"]
            V = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
            N = np.linalg.norm(V, axis=1).astype(np.float32)
            np.save(idx_dir / "semantic_vectors.npy", V)
            np.save(idx_dir / "semantic_norms.npy", N)
            (idx_dir / "semantic_doc_ids.txt").write_text("\n".join(doc_ids) + "\n", encoding="utf-8")
            (idx_dir / "semantic_meta.json").write_text(json.dumps({"model": "dummy", "dimension": 2, "rows": 3}), encoding="utf-8")

            env_old = dict(os.environ)
            try:
                os.environ["SEMANTIC_RETRIEVE_MIX"] = "embed"
                os.environ["VISUAL_RETRIEVE_SUMMARY_ENABLED"] = "0"
                os.environ["VISUAL_RETRIEVE_RETURN_SPANS"] = "1"
                os.environ["RETRIEVE_MIN_TIME_GAP_SEC"] = "0"

                with patch("videoseal.utils.RAG.rag_query_embed.embed_texts", return_value=[[0.0, 1.0]]):
                    out = VisualRetrieveAliasTool().forward(query="cat", top_k=2, index_path=str(idx_dir))
                self.assertIsNone(out.error)
                self.assertIsInstance(out.output, list)
                items = out.output
                self.assertEqual(len(items), 2)
                self.assertEqual(items[0]["start_time"], "00:00:20")
                self.assertEqual(items[0]["end_time"], "00:00:30")
                self.assertIn("blue cat", items[0].get("caption", ""))
            finally:
                os.environ.clear()
                os.environ.update(env_old)

