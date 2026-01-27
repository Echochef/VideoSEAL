import unittest


class TestLVBenchIO(unittest.TestCase):
    def test_parse_choice_letter_basic(self) -> None:
        from videoseal.utils.lvbench_io import parse_choice_letter

        self.assertEqual(parse_choice_letter("<answer> B </answer>"), "B")
        self.assertEqual(parse_choice_letter("<final>(c)</final>"), "C")
        self.assertEqual(parse_choice_letter("Option is: D."), "D")
        self.assertEqual(parse_choice_letter("A"), "A")

    def test_parse_choice_letter_smart_by_option_text(self) -> None:
        from videoseal.utils.lvbench_io import parse_choice_letter_smart

        q = "What color is the scene at the begining?\n(A) White\n(B) Blue\n(C) Brown\n(D) Grey"
        self.assertEqual(parse_choice_letter_smart("The scene looks grey.", q), "D")
        self.assertEqual(parse_choice_letter_smart("It appears blue.", q), "B")

