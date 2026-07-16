import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_models(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return tomllib.load(f)["models"]


class TestRostersStayInSync(unittest.TestCase):
    """configs/models.toml (llama.cpp) and mlx_bench/configs/models.toml
    (MLX) are two separate files by necessity (different `path`s, MLX-only
    `hf_repo` shape — see mlx_bench/README.md) but are meant to cover the
    same models. Nothing enforces that when either file is edited; this is
    that enforcement, run as part of the normal test suite."""

    def setUp(self):
        self.llama_models = load_models(REPO_ROOT / "configs" / "models.toml")
        self.mlx_models = load_models(REPO_ROOT / "mlx_bench" / "configs" / "models.toml")

    def test_same_family_name_pairs_on_both_rosters(self):
        llama_pairs = {(m["family"], m["name"]) for m in self.llama_models}
        mlx_pairs = {(m["family"], m["name"]) for m in self.mlx_models}
        self.assertEqual(
            llama_pairs, mlx_pairs,
            f"llama.cpp-only: {llama_pairs - mlx_pairs}, MLX-only: {mlx_pairs - llama_pairs}",
        )

    def test_every_model_has_an_hf_repo(self):
        for m in self.llama_models + self.mlx_models:
            self.assertIn("hf_repo", m, f"{m['family']}/{m['name']}/{m['quant']} missing hf_repo")

    def test_every_llama_cpp_model_has_hf_files(self):
        for m in self.llama_models:
            self.assertIn("hf_files", m, f"{m['family']}/{m['name']}/{m['quant']} missing hf_files")
            self.assertTrue(m["hf_files"], f"{m['family']}/{m['name']}/{m['quant']} has an empty hf_files list")


if __name__ == "__main__":
    unittest.main()
