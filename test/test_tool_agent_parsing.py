import unittest


class TestToolAgentParsing(unittest.TestCase):
    def test_strip_multiple_choice_options(self) -> None:
        from videoseal.utils.agent.tool_agent_parsing import strip_multiple_choice_options

        q = "What color is the scene?\n(A) White\n(B) Blue\n(C) Brown\n(D) Grey"
        self.assertEqual(strip_multiple_choice_options(q), "What color is the scene?")
        self.assertEqual(strip_multiple_choice_options("No options here."), "No options here.")

