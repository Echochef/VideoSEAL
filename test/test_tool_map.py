import unittest


class TestToolMap(unittest.TestCase):
    def test_only_visual_tools_exposed(self) -> None:
        from videoseal.tools.tool_map import build_tool_map

        m = build_tool_map()
        self.assertEqual(set(m.keys()), {"visual_retrieve", "visual_inspect"})

