import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from coding_eval import extract_code  # noqa: E402


class TestExtractCode(unittest.TestCase):
    def test_extracts_a_single_fenced_python_block(self):
        text = "Here's the code:\n```python\ndef foo():\n    return 1\n```\n"
        self.assertEqual(extract_code(text, "foo"), "def foo():\n    return 1\n")

    def test_extracts_fenced_block_without_a_python_tag(self):
        text = "```\ndef foo():\n    return 1\n```"
        self.assertEqual(extract_code(text, "foo"), "def foo():\n    return 1\n")

    def test_uses_the_last_block_when_there_are_several(self):
        # Thinking-mode responses can contain draft code blocks before the
        # final one — extract_code deliberately takes the last, not first.
        text = (
            "```python\ndef foo():\n    pass  # draft\n```\n"
            "Actually, let me reconsider.\n"
            "```python\ndef foo():\n    return 1\n```"
        )
        self.assertEqual(extract_code(text, "foo"), "def foo():\n    return 1\n")

    def test_returns_none_if_entry_point_is_not_defined(self):
        text = "```python\ndef bar():\n    return 1\n```"
        self.assertIsNone(extract_code(text, "foo"))

    def test_returns_none_when_no_fenced_block_and_no_matching_def(self):
        text = "I can't help with that."
        self.assertIsNone(extract_code(text, "foo"))

    def test_falls_back_to_raw_text_when_unfenced_but_def_present(self):
        text = "def foo():\n    return 1\n"
        self.assertEqual(extract_code(text, "foo"), text)


if __name__ == "__main__":
    unittest.main()
