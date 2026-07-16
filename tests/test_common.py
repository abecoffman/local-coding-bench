import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import find_repo_root, select_models, slugify  # noqa: E402


class TestSlugify(unittest.TestCase):
    def test_joins_parts_with_underscore(self):
        self.assertEqual(slugify("qwen", "Qwen3-4B", "Q4_K_M"), "qwen_Qwen3-4B_Q4_K_M")

    def test_replaces_spaces_with_hyphens(self):
        self.assertEqual(slugify("a b", "c"), "a-b_c")

    def test_drops_empty_parts(self):
        self.assertEqual(slugify("a", "", "b"), "a_b")


class TestSelectModels(unittest.TestCase):
    def setUp(self):
        self.models = [
            {"family": "qwen", "name": "Qwen3-4B-Instruct-2507", "quant": "Q4_K_M"},
            {"family": "qwen", "name": "Qwen3-Coder-Next", "quant": "Q8_0"},
            {"family": "mistral", "name": "Devstral-Small-2-24B-Instruct-2512", "quant": "Q4_K_M"},
        ]

    def test_no_filter_returns_everything(self):
        self.assertEqual(select_models(self.models, None), self.models)
        self.assertEqual(select_models(self.models, []), self.models)

    def test_substring_match_is_case_insensitive(self):
        self.assertEqual(select_models(self.models, ["QWEN3-CODER-NEXT"]), [self.models[1]])

    def test_matches_against_quant_too(self):
        self.assertEqual(select_models(self.models, ["q8_0"]), [self.models[1]])

    def test_multiple_filters_are_ored(self):
        result = select_models(self.models, ["coder-next", "devstral"])
        self.assertEqual(result, [self.models[1], self.models[2]])

    def test_no_match_returns_empty(self):
        self.assertEqual(select_models(self.models, ["nonexistent"]), [])

    def test_setup_sh_shorthand_tokens_match_real_entries(self):
        # setup.sh's --models tokens (e.g. "llama-3.1-8b") are hand-picked
        # shorthands, not exact family/name values — this substring contract
        # is what setup.sh and download_models.py actually rely on.
        models = [{"family": "llama", "name": "Meta-Llama-3.1-8B-Instruct", "quant": "Q4_K_M"}]
        self.assertEqual(select_models(models, ["llama-3.1-8b"]), models)


class TestFindRepoRoot(unittest.TestCase):
    def test_finds_root_from_a_nested_start(self):
        root = find_repo_root(Path(__file__).resolve().parent)  # tests/
        self.assertTrue((root / "CONTRIBUTING.md").exists())

    def test_finds_root_when_start_is_the_root_itself(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(find_repo_root(repo_root), repo_root)

    def test_raises_when_no_marker_is_found(self):
        with self.assertRaises(RuntimeError):
            find_repo_root(Path("/"))


if __name__ == "__main__":
    unittest.main()
