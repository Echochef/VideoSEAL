import tempfile
import unittest
from pathlib import Path


class TestRunFromParquetCLI(unittest.TestCase):
    def test_parse_shell_env_file(self) -> None:
        from videoseal.cli.run_from_parquet import parse_shell_env_file

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.env"
            p.write_text("A=1\nB='two'\nC=hello\\ world\n", encoding="utf-8")
            out = parse_shell_env_file(p)
            self.assertEqual(out["A"], "1")
            self.assertEqual(out["B"], "two")
            self.assertEqual(out["C"], "hello world")

    def test_resolve_vllm_discovery_from_meta(self) -> None:
        from videoseal.cli.run_from_parquet import resolve_vllm_discovery

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            meta = repo / "serve" / "vllm_servers.env"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(
                "\n".join(
                    [
                        "VLLM_HOST=0.0.0.0",
                        "VLLM_BASE_PORT=20000",
                        "VLLM_NUM_SERVERS=2",
                        "VLLM_SERVED_MODEL_NAME=Qwen3-8B",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            env = {
                "USE_VLLM_META_FILE": "1",
                "VLLM_META_FILE": meta.as_posix(),
                "VLLM_META_REQUIRE_MODEL_MATCH": "1",
                "VLLM_MODEL": "Qwen/Qwen3-8B",
            }
            vllm = resolve_vllm_discovery(env, repo_root=repo)
            self.assertTrue(vllm.meta_used)
            self.assertEqual(vllm.host, "127.0.0.1")  # 0.0.0.0 normalized for client
            self.assertEqual(vllm.base_port, 20000)
            self.assertEqual(vllm.num_servers, 2)
            self.assertEqual(vllm.openai_base_for_shard(0), "http://127.0.0.1:20000/v1")
            self.assertEqual(vllm.openai_base_for_shard(1), "http://127.0.0.1:20001/v1")

    def test_resolve_visual_inspect_base_pool(self) -> None:
        from videoseal.cli.run_from_parquet import resolve_visual_inspect_base_pool

        env = {"VISUAL_INSPECT_API_BASES": "http://a/v1, http://b/v1"}
        self.assertEqual(resolve_visual_inspect_base_pool(env), ["http://a/v1", "http://b/v1"])

        env2 = {"TOOL_VLM_HOST": "0.0.0.0", "TOOL_VLM_PORTS": "19001,19002"}
        self.assertEqual(resolve_visual_inspect_base_pool(env2), ["http://127.0.0.1:19001/v1", "http://127.0.0.1:19002/v1"])

    def test_resolve_run_num_shards_and_concurrency(self) -> None:
        from videoseal.cli.run_from_parquet import resolve_concurrency, resolve_run_num_shards

        env = {}
        shards = resolve_run_num_shards(vllm_num_servers=2, tool_pool_size=5, env=env)
        self.assertEqual(shards, 5)

        env_force = {"FORCE_NUM_SHARDS": "3"}
        shards2 = resolve_run_num_shards(vllm_num_servers=2, tool_pool_size=5, env=env_force)
        self.assertEqual(shards2, 3)

        env_conc = {"CONCURRENCY": "64"}
        plan = resolve_concurrency(env=env_conc, num_shards=3)
        self.assertEqual(plan.per_shard, 22)
        self.assertEqual(plan.total, 66)

        env_conc0 = {"CONCURRENCY": "0"}
        plan0 = resolve_concurrency(env=env_conc0, num_shards=3)
        self.assertEqual(plan0.per_shard, 1)
        self.assertEqual(plan0.total, 3)


if __name__ == "__main__":
    unittest.main()

