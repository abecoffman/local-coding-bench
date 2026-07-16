import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate import aggregate_rows  # noqa: E402


def make_row(**overrides) -> dict:
    row = {
        "model_family": "qwen",
        "model_name": "Qwen3-4B-Instruct-2507",
        "quant": "Q4_K_M",
        "runtime": "llama.cpp",
        "n_gpu_layers": 99,
        "n_prompt": 512,
        "n_gen": 0,
        "n_depth": 0,
        "avg_ts": 100.0,
        "config": "standard",
    }
    row.update(overrides)
    return row


class TestAggregateRows(unittest.TestCase):
    def test_groups_repeated_runs_of_the_same_test_point(self):
        rows = [make_row(avg_ts=100.0), make_row(avg_ts=110.0)]
        [agg] = aggregate_rows(rows)
        self.assertEqual(agg["n_runs"], 2)
        self.assertEqual(agg["avg_ts_mean"], 105.0)
        self.assertEqual(agg["avg_ts_samples"], [100.0, 110.0])

    def test_stddev_is_zero_for_a_single_run(self):
        [agg] = aggregate_rows([make_row()])
        self.assertEqual(agg["avg_ts_stddev"], 0.0)

    def test_groups_by_n_gpu_layers_not_by_config(self):
        # A CPU-only (-ngl 0) run and a GPU run of the same model/test-shape
        # are genuinely different measurements and must not merge, even
        # though they're the "same" test point in every other field.
        rows = [
            make_row(n_gpu_layers=99, avg_ts=100.0, config="standard"),
            make_row(n_gpu_layers=0, avg_ts=5.0, config="cpu-baseline"),
        ]
        self.assertEqual(len(aggregate_rows(rows)), 2)

    def test_ignores_config_field_when_grouping_the_same_test_point(self):
        # A sweep rename (or a rerun logged under a different config name)
        # shouldn't fragment what's actually the same measurement.
        rows = [make_row(avg_ts=100.0, config="standard"), make_row(avg_ts=100.0, config="_rerun-standard")]
        [agg] = aggregate_rows(rows)
        self.assertEqual(agg["n_runs"], 2)

    def test_test_label_for_prompt_processing(self):
        [agg] = aggregate_rows([make_row(n_prompt=512, n_gen=0, n_depth=0)])
        self.assertEqual(agg["test"], "pp512")

    def test_test_label_for_generation(self):
        [agg] = aggregate_rows([make_row(n_prompt=0, n_gen=128, n_depth=0)])
        self.assertEqual(agg["test"], "tg128")

    def test_test_label_includes_depth_when_nonzero(self):
        [agg] = aggregate_rows([make_row(n_prompt=512, n_gen=0, n_depth=4096)])
        self.assertEqual(agg["test"], "pp512@d4096")

    def test_different_quants_stay_separate(self):
        rows = [make_row(quant="Q4_K_M"), make_row(quant="Q8_0")]
        self.assertEqual(len(aggregate_rows(rows)), 2)


if __name__ == "__main__":
    unittest.main()
