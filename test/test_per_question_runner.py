import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class TestPerQuestionRunner(unittest.TestCase):
    def test_map_step1_dir_to_unified_semantic(self) -> None:
        from videoseal.runner.per_question_runner import _map_step1_dir_to_unified_semantic

        p = "/mnt/shanhai-ai/qiuchenhao/data/LVU/data/step1/LVBench/vid123"
        out = _map_step1_dir_to_unified_semantic(p)
        self.assertEqual(out, "/mnt/shanhai-ai/qiuchenhao/data/LVU/data/indexes/semantic/LVBench/vid123")

        self.assertIsNone(_map_step1_dir_to_unified_semantic("/a/b/c/not_step1/LVBench/vid123"))
        self.assertIsNone(_map_step1_dir_to_unified_semantic("/mnt/shanhai-ai/qiuchenhao/data/LVU/data/step1/only_one_part"))

    def test_meta_json_benchmark_strict(self) -> None:
        from videoseal.runner.per_question_runner import _meta_json_benchmark

        self.assertEqual(_meta_json_benchmark({"meta_json": {"benchmark": "LVBench"}}), "LVBench")
        self.assertEqual(_meta_json_benchmark({"meta_json": json.dumps({"benchmark": "LVBench"})}), "LVBench")
        self.assertIsNone(_meta_json_benchmark({"meta_json": {"benchmark": ""}}))

        with self.assertRaises(TypeError):
            _meta_json_benchmark({"meta_json": "[]"})

        with self.assertRaises(TypeError):
            _meta_json_benchmark({"meta_json": []})  # type: ignore[arg-type]

    def test_optional_positive_float(self) -> None:
        from videoseal.runner.per_question_runner import _optional_positive_float

        self.assertIsNone(_optional_positive_float(None))
        self.assertIsNone(_optional_positive_float(""))
        self.assertIsNone(_optional_positive_float("0"))
        self.assertIsNone(_optional_positive_float("-1"))
        self.assertEqual(_optional_positive_float("2.5"), 2.5)

        with self.assertRaises(ValueError):
            _optional_positive_float("not-a-number")

    def test_load_tasks_from_parquet_maps_step1_semantic(self) -> None:
        from videoseal.runner.per_question_runner import load_tasks_from_parquet

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "toy.parquet"
            extra = {
                "video_id": "vid1",
                "qa_uid": "u1",
                "question": "Q?",
                "ground_truth": "A",
                "VIDEO_PATH": "/tmp/vid1.mp4",
                "SEMANTIC_INDEX_DIR": "/mnt/shanhai-ai/qiuchenhao/data/LVU/data/step1/LVBench/vid1",
                "meta_json": json.dumps({"benchmark": "LVBench"}),
            }
            table = pa.Table.from_pydict({"extra_info": [extra]})
            pq.write_table(table, p)

            tasks = load_tasks_from_parquet(p)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["video_id"], "vid1")
            self.assertEqual(tasks[0]["uid"], "u1")
            self.assertEqual(tasks[0]["benchmark"], "LVBench")
            self.assertEqual(
                tasks[0]["visual_index"],
                "/mnt/shanhai-ai/qiuchenhao/data/LVU/data/indexes/semantic/LVBench/vid1",
            )


if __name__ == "__main__":
    unittest.main()

