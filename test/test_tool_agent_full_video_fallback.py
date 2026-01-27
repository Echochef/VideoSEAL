import os
import unittest


class _Env:
    def __init__(self, updates: dict[str, str | None]):
        self._updates = dict(updates)
        self._prev: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._updates.items():
            self._prev[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class TestFullVideoVisualInspectFallback(unittest.TestCase):
    def test_forced_full_video_visual_inspect_fallback_uses_0_to_duration(self):
        from videoseal.agents.tool_agent import SimpleToolAgent
        from videoseal.tools.base import Tool, ToolOutput

        calls: list[dict] = []

        class DummyInspect(Tool):
            def __init__(self):
                super().__init__(name="visual_inspect", description="dummy")

            @property
            def json(self) -> dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "visual_inspect",
                        "description": "dummy",
                        "parameters": {"type": "object", "properties": {"spans": {"type": "array"}}, "required": ["spans"]},
                    },
                }

            def forward(self, *, video_path: str, spans, questions, context=None, prompt_type: int = 0) -> ToolOutput:
                calls.append(
                    {
                        "video_path": video_path,
                        "spans": spans,
                        "questions": questions,
                        "prompt_mode": os.getenv("VISUAL_INSPECT_PROMPT_MODE"),
                        "inspect_fps": os.getenv("INSPECT_FPS"),
                    }
                )
                return ToolOutput(self.name, output={"answer": "<answer>D</answer>"})

        tools = {"visual_inspect": DummyInspect}

        with _Env(
            {
                "AGENT_LLM_BACKEND": "api",
                "AGENT_API_USE_MESSAGES": "1",
                "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "1",
                "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE": "mcq",
                "AGENT_LLM_MODEL": "dummy",
                "AGENT_LLM_API_BASE": "http://127.0.0.1:1/v1",
                "AGENT_LLM_API_KEY": "dummy",
                "AGENT_LLM_MAX_TOKENS": "16",
                "AGENT_LLM_TEMPERATURE": "0",
                "AGENT_LLM_TIMEOUT": "1",
                "AGENT_MLLM_BACKEND": "openai",
                "VIDEO_DURATION_SEC": "123",
                "INSPECT_MAX_TOTAL_IMAGES": "10",
                "VISUAL_INSPECT_PROMPT_MODE": "raw",
                "INSPECT_FPS": "2",
            }
        ):
            agent = SimpleToolAgent(tools=tools)

            def _dummy_generate_chat(*_a, **_kw):
                return "<thinking>call tool</thinking>"

            agent.llm.generate_chat = _dummy_generate_chat  # type: ignore[method-assign]
            agent.llm.get_last_usage = lambda: None  # type: ignore[method-assign]

            tool_calls = [{"id": "x", "function": {"name": "unknown_tool", "arguments": {}}}]
            agent._extract_tool_calls_from_last_response = lambda: ({"content": ""}, tool_calls)  # type: ignore[method-assign]

            res = agent.run(
                question="What is the answer? (A) foo (B) bar (C) baz (D) qux",
                uid="u",
                video_id="vid",
                video_path="/tmp/does_not_need_to_exist.mp4",
                visual_index=None,
                groundtruth="D",
                max_steps=2,
                save_dir=None,
                prompt_type=0,
            )

            self.assertEqual(res.get("final"), "D")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["spans"], [{"start_time": "00:00:00", "end_time": "00:02:03"}])
            self.assertEqual(calls[0]["prompt_mode"], "mcq")
            self.assertIsNotNone(calls[0]["inspect_fps"])
            self.assertAlmostEqual(float(calls[0]["inspect_fps"]), 10.0 / 123.0, places=6)

            # Env restored after forced fallback.
            self.assertEqual(os.getenv("VISUAL_INSPECT_PROMPT_MODE"), "raw")
            self.assertEqual(os.getenv("INSPECT_FPS"), "2")

