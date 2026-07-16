#!/usr/bin/env python3
"""Aggregate repeated throughput runs in results.jsonl into mean/stddev per
unique test point (same model/quant/runtime/hardware-mode/test-shape,
however many times and under whatever sweep name it was run under). Doesn't
mutate results.jsonl — writes a derived file.

Deliberately groups by n_gpu_layers, not by `config` (the sweep name that
produced a row): a rename of standard.toml (or a rerun under a differently
named config) shouldn't fragment what's actually the same measurement into
separate groups, and n_gpu_layers is what actually distinguishes a
meaningfully different test (CPU-only vs GPU-offloaded), not the label of
whatever config file happened to produce it.

Usage:
    python3 scripts/aggregate.py [results.jsonl] [-o results-aggregated.jsonl] [--csv results-aggregated.csv]

The .jsonl output keeps the full per-run sample list for anyone who wants
it; the .csv is the flat mean/stddev table meant for skimming — GitHub
renders .csv as an actual table in its web UI, unlike .jsonl.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

GROUP_KEYS = ["model_family", "model_name", "quant", "runtime", "n_gpu_layers", "n_prompt", "n_gen", "n_depth"]


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Group raw results.jsonl rows by GROUP_KEYS into one mean/stddev/n
    entry per unique test point, sorted for stable output, each tagged with
    a human-readable `test` label (e.g. "pp512", "tg128@d4096"). Pure
    function, no file I/O, so it's testable without touching disk."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in GROUP_KEYS)
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, group_rows in groups.items():
        samples = [r["avg_ts"] for r in group_rows]
        agg = dict(zip(GROUP_KEYS, key))
        agg["avg_ts_mean"] = statistics.mean(samples)
        agg["avg_ts_stddev"] = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        agg["n_runs"] = len(samples)
        agg["avg_ts_samples"] = samples
        aggregated.append(agg)

    aggregated.sort(key=lambda a: (a["model_family"], a["model_name"], a["quant"], a["runtime"], a["n_depth"] or 0))

    for a in aggregated:
        test = f"pp{a['n_prompt']}" if a["n_prompt"] else f"tg{a['n_gen']}"
        if a["n_depth"]:
            test += f"@d{a['n_depth']}"
        a["test"] = test

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path("results.jsonl"))
    parser.add_argument("-o", "--output", type=Path, default=None, help="write full aggregated .jsonl (with sample lists)")
    parser.add_argument("--csv", type=Path, default=None, help="write a flat mean/stddev table as .csv")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.results.open()]
    aggregated = aggregate_rows(rows)

    if args.output:
        with args.output.open("w") as f:
            for a in aggregated:
                f.write(json.dumps(a) + "\n")
        print(f"wrote {len(aggregated)} aggregated rows to {args.output}")

    if args.csv:
        csv_columns = ["model_family", "model_name", "quant", "runtime", "n_gpu_layers", "test", "n_runs", "avg_ts_mean", "avg_ts_stddev"]
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
            writer.writeheader()
            for a in aggregated:
                row = dict(a)
                row["avg_ts_mean"] = round(row["avg_ts_mean"], 2)
                row["avg_ts_stddev"] = round(row["avg_ts_stddev"], 2)
                writer.writerow(row)
        print(f"wrote {len(aggregated)} rows to {args.csv}")

    header = f"{'model':<28}{'quant':<10}{'runtime':<10}{'ngl':<6}{'test':<14}{'n':<4}{'mean tok/s':<14}{'stddev':<10}"
    print(header)
    print("-" * len(header))
    for a in aggregated:
        ngl = a["n_gpu_layers"] if a["n_gpu_layers"] is not None else "-"
        print(
            f"{a['model_name']:<28}{a['quant']:<10}{a['runtime']:<10}{str(ngl):<6}{a['test']:<14}"
            f"{a['n_runs']:<4}{a['avg_ts_mean']:<14.2f}{a['avg_ts_stddev']:<10.2f}"
        )


if __name__ == "__main__":
    main()
