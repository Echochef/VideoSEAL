import argparse
import os
import unittest


class TestVllmOpenAIWrapper(unittest.TestCase):
    def test_build_server_cmd_extra_args(self) -> None:
        from videoseal.serve.vllm_openai import build_server_cmd

        old = os.environ.get("VLLM_EXTRA_ARGS")
        try:
            os.environ["VLLM_EXTRA_ARGS"] = "--max-model-len 4096 --dtype bfloat16"
            ns = argparse.Namespace(
                model="/tmp/model",
                served_model_name="served",
                host="127.0.0.1",
                port=18080,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )
            cmd = build_server_cmd(ns)
            self.assertIn("--max-model-len", cmd)
            self.assertIn("4096", cmd)
            self.assertIn("--dtype", cmd)
            self.assertIn("bfloat16", cmd)
        finally:
            if old is None:
                os.environ.pop("VLLM_EXTRA_ARGS", None)
            else:
                os.environ["VLLM_EXTRA_ARGS"] = old

