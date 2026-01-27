import os
import unittest


class TestEnvStrict(unittest.TestCase):
    def test_env_int_strict_default_and_parse(self) -> None:
        from videoseal.utils.agent.env import env_int_strict

        env_old = dict(os.environ)
        try:
            os.environ.pop("X_INT", None)
            self.assertEqual(env_int_strict("X_INT", 7), 7)

            os.environ["X_INT"] = "9"
            self.assertEqual(env_int_strict("X_INT", 7), 9)

            os.environ["X_INT"] = "bad"
            with self.assertRaises(RuntimeError):
                env_int_strict("X_INT", 7)
        finally:
            os.environ.clear()
            os.environ.update(env_old)

    def test_env_float_strict_default_and_parse(self) -> None:
        from videoseal.utils.agent.env import env_float_strict

        env_old = dict(os.environ)
        try:
            os.environ.pop("X_FLOAT", None)
            self.assertEqual(env_float_strict("X_FLOAT", 1.5), 1.5)

            os.environ["X_FLOAT"] = "2.25"
            self.assertEqual(env_float_strict("X_FLOAT", 1.5), 2.25)

            os.environ["X_FLOAT"] = "bad"
            with self.assertRaises(RuntimeError):
                env_float_strict("X_FLOAT", 1.5)
        finally:
            os.environ.clear()
            os.environ.update(env_old)

    def test_env_int_first_prefers_first_present(self) -> None:
        from videoseal.utils.agent.env import env_int_first

        env_old = dict(os.environ)
        try:
            os.environ.pop("A_INT", None)
            os.environ.pop("B_INT", None)
            self.assertEqual(env_int_first(("A_INT", "B_INT"), 3), 3)

            os.environ["B_INT"] = "5"
            self.assertEqual(env_int_first(("A_INT", "B_INT"), 3), 5)

            os.environ["A_INT"] = "7"
            self.assertEqual(env_int_first(("A_INT", "B_INT"), 3), 7)
        finally:
            os.environ.clear()
            os.environ.update(env_old)

    def test_env_float_first_prefers_first_present(self) -> None:
        from videoseal.utils.agent.env import env_float_first

        env_old = dict(os.environ)
        try:
            os.environ.pop("A_FLOAT", None)
            os.environ.pop("B_FLOAT", None)
            self.assertEqual(env_float_first(("A_FLOAT", "B_FLOAT"), 3.0), 3.0)

            os.environ["B_FLOAT"] = "5.5"
            self.assertEqual(env_float_first(("A_FLOAT", "B_FLOAT"), 3.0), 5.5)

            os.environ["A_FLOAT"] = "7.25"
            self.assertEqual(env_float_first(("A_FLOAT", "B_FLOAT"), 3.0), 7.25)
        finally:
            os.environ.clear()
            os.environ.update(env_old)


if __name__ == "__main__":
    unittest.main()
