#!/usr/bin/env python3
"""Run a llama-bench sweep across a model roster, appending normalized rows to
results.jsonl. Models and sweep parameters are separate files so adding a
model doesn't mean editing every sweep, and vice versa.

Usage:
    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml [--label m5-max-macbook]
    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --model qwen3-coder-next
"""
import argparse
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from _common import find_repo_root, select_models, slugify

REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
RESULTS_PATH = REPO_ROOT / "results.jsonl"
RUNS_DIR = REPO_ROOT / "runs"
BINARY = REPO_ROOT / "runtimes" / "llama.cpp" / "build" / "bin" / "llama-bench"

# llama-bench's own JSONL output has ~50 fields; most are its fixed CLI
# defaults that never vary across our sweeps (thread count, batch size, KV
# cache dtype, ...) and would just be repeated verbatim on every row. Keep
# only what's interpretable or meant to vary (across sweeps, models, or
# future contributors' machines) — the full unfiltered output is still in
# runs/<run_id>/stdout.jsonl for anyone who needs it.
KEEP_FIELDS = {
    "model_type", "model_size", "model_n_params",
    "n_prompt", "n_gen", "n_depth", "n_gpu_layers",
    "test_time", "avg_ts", "stddev_ts", "samples_ts",
    "cpu_info", "gpu_info", "build_commit",
}


def flatten_args(args: dict) -> list[str]:
    flat = []
    for key, value in args.items():
        flat.append(key)
        flat.append(str(value))
    return flat


def run_model(model: dict, extra_args: list[str], sweep_name: str, label: str) -> int:
    model_path = REPO_ROOT / model["path"]
    if not model_path.exists():
        print(f"skip: model file not found: {model_path}", file=sys.stderr)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{slugify(model['family'], model['name'], model['quant'])}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(BINARY), "-m", str(model_path), "-o", "jsonl", *extra_args]
    print(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    (run_dir / "stdout.jsonl").write_text(proc.stdout)
    (run_dir / "stderr.log").write_text(proc.stderr)
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")

    if proc.returncode != 0:
        print(f"FAILED (exit {proc.returncode}), see {run_dir / 'stderr.log'}", file=sys.stderr)
        return 0

    count = 0
    with RESULTS_PATH.open("a") as results_file:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            full_row = json.loads(line)
            row = {k: v for k, v in full_row.items() if k in KEEP_FIELDS}
            row.update(
                {
                    "runtime": "llama.cpp",
                    "run_id": run_id,
                    "config": sweep_name,
                    "label": label,
                    "model_family": model["family"],
                    "model_name": model["name"],
                    "quant": model["quant"],
                }
            )
            results_file.write(json.dumps(row) + "\n")
            count += 1

    print(f"  -> {count} result row(s) appended, raw output in {run_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="path to a model roster .toml file")
    parser.add_argument("--sweep", type=Path, required=True, help="path to a sweep-args .toml file")
    parser.add_argument("--label", default="", help="machine/environment tag (e.g. m5-max-macbook)")
    parser.add_argument(
        "--model",
        action="append",
        help="substring filter on 'family name quant' (case-insensitive), e.g. --model qwen3-coder-next. "
        "Repeatable, OR'd together. Omit to run the whole roster.",
    )
    args = parser.parse_args()

    with args.models.open("rb") as f:
        models_config = tomllib.load(f)
    with args.sweep.open("rb") as f:
        sweep_config = tomllib.load(f)

    runtime = models_config.get("runtime", "llama.cpp")
    if runtime != "llama.cpp":
        sys.exit(f"{args.models} declares runtime={runtime!r}, but bench.py only drives llama.cpp")

    if not BINARY.exists():
        sys.exit(f"binary not found: {BINARY} (did you run setup.sh?)")

    extra_args = flatten_args(sweep_config.get("args", {}))
    sweep_name = args.sweep.stem

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    models = select_models(models_config["models"], args.model)
    if not models:
        sys.exit(f"--model {args.model} matched no models in {args.models}")

    total = 0
    for model in models:
        total += run_model(model, extra_args, sweep_name, args.label)

    print(f"done: {total} total result row(s)")


if __name__ == "__main__":
    main()
