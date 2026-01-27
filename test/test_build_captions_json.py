import unittest


class TestBuildCaptionsJson(unittest.TestCase):
    def test_slugify(self) -> None:
        from videoseal.utils.mllm.build_captions_json import _slugify

        self.assertEqual(_slugify("Hello World"), "hello-world")
        self.assertEqual(_slugify("  a__b  "), "a-b")

    def test_is_valid_cached_caption(self) -> None:
        from videoseal.utils.mllm.build_captions_json import _is_valid_cached_caption

        self.assertTrue(_is_valid_cached_caption({"clip_description": "ok"}, retry_error_caps=False))
        self.assertFalse(_is_valid_cached_caption({"clip_description": ""}, retry_error_caps=False))
        self.assertFalse(_is_valid_cached_caption({}, retry_error_caps=False))
        self.assertFalse(_is_valid_cached_caption({"clip_description": "[ERROR] Too Many Requests"}, retry_error_caps=False))
        self.assertTrue(_is_valid_cached_caption({"clip_description": "[ERROR] No frames sampled"}, retry_error_caps=False))
        self.assertFalse(_is_valid_cached_caption({"clip_description": "[ERROR] No frames sampled"}, retry_error_caps=True))

    def test_normalize_entities(self) -> None:
        from videoseal.utils.mllm.build_captions_json import _normalize_entities

        cd = {
            "entities": [
                {"type": "person", "text": "Alice"},
                {"name": "Bob"},
                "Charlie",
            ],
            "objects": [{"name": "Car", "attributes": ["red", "fast", "red"]}],
            "actions": [{"subject": "Alice", "verb": "drives", "object": "Car"}],
            "scene": {"location": "street", "lighting": "day", "camera": "closeup"},
            "text_in_frame": ["HELLO", "HELLO", ""],
            "keywords": ["k1"],
            "subject_registry": {"Eve": {}},
        }

        ents = _normalize_entities(cd)
        self.assertIn({"type": "person", "text": "Alice"}, ents)
        self.assertIn({"type": "entity", "text": "Bob"}, ents)
        self.assertIn({"type": "entity", "text": "Charlie"}, ents)
        self.assertIn({"type": "object", "text": "Car"}, ents)
        self.assertIn({"type": "attribute", "text": "red"}, ents)
        self.assertIn({"type": "attribute", "text": "fast"}, ents)
        self.assertIn({"type": "subject", "text": "Alice"}, ents)
        self.assertIn({"type": "action", "text": "drives"}, ents)
        self.assertIn({"type": "object", "text": "Car"}, ents)
        self.assertIn({"type": "scene", "text": "street"}, ents)
        self.assertIn({"type": "scene", "text": "day"}, ents)
        self.assertIn({"type": "scene", "text": "closeup"}, ents)
        self.assertIn({"type": "text", "text": "HELLO"}, ents)
        self.assertIn({"type": "keyword", "text": "k1"}, ents)
        self.assertIn({"type": "person", "text": "Eve"}, ents)

