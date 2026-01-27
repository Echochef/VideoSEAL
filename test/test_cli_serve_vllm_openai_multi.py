import unittest


class TestServeVllmOpenAIMultiCLI(unittest.TestCase):
    def test_parse_gpu_ids(self) -> None:
        from videoseal.cli.serve_vllm_openai_multi import parse_gpu_ids

        self.assertEqual(parse_gpu_ids("0, 1,2"), ["0", "1", "2"])
        self.assertEqual(parse_gpu_ids(""), [])

    def test_render_meta_env(self) -> None:
        from videoseal.cli.serve_vllm_openai_multi import render_meta_env

        text = render_meta_env(
            host="127.0.0.1",
            base_port=18080,
            num_servers=2,
            model="Qwen/Qwen3-8B",
            served_model_name="Qwen3-8B",
        )
        self.assertIn("VLLM_HOST=", text)
        self.assertIn("VLLM_BASE_PORT=", text)
        self.assertIn("VLLM_NUM_SERVERS=", text)
        self.assertIn("VLLM_MODEL=", text)
        self.assertIn("VLLM_SERVED_MODEL_NAME=", text)


if __name__ == "__main__":
    unittest.main()

