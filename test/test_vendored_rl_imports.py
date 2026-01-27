import unittest


class TestVendoredRlImports(unittest.TestCase):
    def test_import_rllm_and_verl(self) -> None:
        import rllm  # noqa: F401
        import verl  # noqa: F401

        from rllm.agents.tool_agent import ToolAgent  # noqa: F401
        from rllm.environments.tools.tool_env import ToolEnvironment  # noqa: F401
        from rllm.rewards.video_reward_dp import reward_video  # noqa: F401
        from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow  # noqa: F401

    def test_rllm_video_tools_are_minimal_and_match_separator(self) -> None:
        import os

        os.environ.setdefault("AGENT_SYS_PROMPT_LANG", "en")

        import rllm.tools as tools

        from rllm.agents.video_system_prompts import TOOLAGENT_SYS_PROMPT
        from rllm.tools.video_tools import VisualInspectTool, VisualRetrieveTool
        from videoseal.agents.tool_agent import _system_prompt as separator_system_prompt
        from videoseal.tools.visual_tools import VisualInspectAliasTool, VisualRetrieveAliasTool

        # Prompt is shared with Videoseal runner (scripts/run_vllm_from_jsonl.sh).
        self.assertEqual(TOOLAGENT_SYS_PROMPT, separator_system_prompt("en").strip())

        # Only keep the two video tools; other legacy names must not be registered by default.
        self.assertIsNotNone(tools.tool_registry.instantiate("visual_retrieve"))
        self.assertIsNotNone(tools.tool_registry.instantiate("visual_inspect"))
        self.assertIsNone(tools.tool_registry.instantiate("image_retrieve"))
        self.assertIsNone(tools.tool_registry.instantiate("answer_from_spans"))
        self.assertIsNone(tools.tool_registry.instantiate("overview_read"))
        self.assertIsNone(tools.tool_registry.instantiate("read_summary"))

        # Tool schemas should be identical to Videoseal's canonical tools.
        self.assertEqual(VisualRetrieveTool().json, VisualRetrieveAliasTool().json)
        self.assertEqual(VisualInspectTool().json, VisualInspectAliasTool().json)
