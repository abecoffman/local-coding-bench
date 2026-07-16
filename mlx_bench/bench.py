#!/usr/bin/env python3
"""Runs an MLX benchmark sweep and appends normalized rows to a results file,
matching the row schema a llama.cpp-style benchmark driver (e.g. one built
around llama-bench) would use.

Depth methodology: for a given depth D, we build a fresh prompt_cache, prefill
it with D filler tokens (untimed), then measure against that primed cache:
  - pp row: process n_prompt new tokens against the D-token cache, max_tokens=1.
    prompt_tps is isolated to just the new tokens, matching llama-bench's `-d`.
  - tg row: generate n_gen tokens starting from the D-token cache.
    generation_tps matches llama-bench's tg-at-depth methodology.
This isolation matters: measuring pp as a single depth+n_prompt prefill
without cache reuse would conflate attention-over-cache cost with raw batch
throughput, which increases with batch size and trends the wrong direction.

Every path (--models, --sweep, a model's `path` in the roster, --output,
--runs-dir) is resolved relative to wherever you run this from — nothing
here looks outside this invocation's own arguments for where things live.
--output/--runs-dir default to ./results.jsonl and ./runs, so running from
the repo root (as in the first two examples below) lands them there by
default; running from within this directory keeps everything local to it.

Usage:
    .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml [--label m5-max-macbook]
    .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml --model qwen3-coder-next
    cd mlx_bench && .venv/bin/python bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml  # fully standalone: results.jsonl/runs/ stay in this directory
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

FILLER_TEXT = "The quick brown fox jumps over the lazy dog. " * 4000


def slugify(*parts: str) -> str:
    return "_".join(p.replace(" ", "-") for p in parts if p)


def select_models(models: list[dict], filters: list[str] | None) -> list[dict]:
    """Filter a model roster by --model substrings, OR'd together, matched
    case-insensitively against "family name quant". No filters means the
    whole roster, preserving the old default behavior."""
    if not filters:
        return models
    needles = [f.lower() for f in filters]
    return [
        m for m in models
        if any(n in f"{m['family']} {m['name']} {m['quant']}".lower() for n in needles)
    ]


def model_ready(model_path: Path) -> bool:
    """True if model_path is a fully-downloaded MLX model dir. A bare
    .exists() is not enough: hf download creates the directory immediately,
    so a still-downloading model would otherwise look "ready" and crash
    mid-run on missing safetensors."""
    if not model_path.is_dir():
        return False
    if not (model_path / "config.json").exists():
        return False
    if not any(model_path.glob("*.safetensors")):
        return False
    if any(model_path.rglob("*.incomplete")):
        return False
    return True


def gpu_info() -> str:
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "Chip:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


_filler_ids_cache: list[int] | None = None


def filler_tokens(tokenizer, n: int) -> list[int]:
    global _filler_ids_cache
    if _filler_ids_cache is None:
        _filler_ids_cache = tokenizer.encode(FILLER_TEXT)
    ids = _filler_ids_cache
    while len(ids) < n:
        ids = ids + ids
    return ids[:n] if n > 0 else ids[:1]


def timed_generate(model, tokenizer, prompt_ids, max_tokens, prompt_cache=None):
    response = None
    for response in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=max_tokens, prompt_cache=prompt_cache
    ):
        pass
    return response


def primed_cache(model, tokenizer, depth):
    pcache = make_prompt_cache(model)
    if depth > 0:
        timed_generate(model, tokenizer, filler_tokens(tokenizer, depth), max_tokens=1, prompt_cache=pcache)
    return pcache


def run_point(model, tokenizer, n_prompt, n_gen, n_depth, reps):
    # +1 and drop the first sample: MLX can pay a one-time cost (graph
    # construction, buffer allocation) the first time it sees a given
    # prompt/generation shape, which would otherwise skew small-rep averages.
    pp_ts, tg_ts = [], []
    for i in range(reps + 1):
        if n_prompt:
            pcache = primed_cache(model, tokenizer, n_depth)
            pp_prompt = filler_tokens(tokenizer, n_prompt)
            resp = timed_generate(model, tokenizer, pp_prompt, max_tokens=1, prompt_cache=pcache)
            if i > 0:
                pp_ts.append(resp.prompt_tps)
        if n_gen:
            pcache = primed_cache(model, tokenizer, n_depth)
            tg_prompt = filler_tokens(tokenizer, 1)
            resp = timed_generate(model, tokenizer, tg_prompt, max_tokens=n_gen, prompt_cache=pcache)
            if i > 0:
                tg_ts.append(resp.generation_tps)
    return pp_ts, tg_ts


def run_model(model_cfg: dict, args: dict, sweep_name: str, label: str, results_path: Path, runs_dir: Path) -> int:
    model_path = Path(model_cfg["path"])
    if not model_ready(model_path):
        print(f"skip: model not ready (missing or still downloading): {model_path}", file=sys.stderr)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{slugify(model_cfg['family'], model_cfg['name'], model_cfg['quant'])}_mlx"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading: {model_path}")
    model, tokenizer = load(str(model_path))

    print("warming up...")
    timed_generate(model, tokenizer, filler_tokens(tokenizer, 16), max_tokens=8)

    p_values = args.get("p", [512])
    n_gen = args.get("n", 128)
    d_values = args.get("d", [0])
    reps = args.get("r", 5)
    gpu = gpu_info()

    count = 0
    log_lines = []
    with results_path.open("a") as results_file:
        for depth in d_values:
            for p in p_values:
                pp_ts, tg_ts = run_point(model, tokenizer, p, n_gen, depth, reps)

                for kind, samples, n_prompt_val, n_gen_val in (
                    ("pp", pp_ts, p, 0),
                    ("tg", tg_ts, 0, n_gen),
                ):
                    if not samples:
                        continue
                    row = {
                        "runtime": "mlx",
                        "mlx_lm_version": mlx_lm.__version__,
                        "gpu_info": gpu,
                        "n_prompt": n_prompt_val,
                        "n_gen": n_gen_val,
                        "n_depth": depth,
                        "test_time": datetime.now(timezone.utc).isoformat(),
                        "avg_ts": statistics.mean(samples),
                        "stddev_ts": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
                        "samples_ts": samples,
                        "run_id": run_id,
                        "config": sweep_name,
                        "label": label,
                        "model_family": model_cfg["family"],
                        "model_name": model_cfg["name"],
                        "quant": model_cfg["quant"],
                    }
                    results_file.write(json.dumps(row) + "\n")
                    log_lines.append(json.dumps(row))
                    count += 1
                    print(
                        f"  depth={depth} {kind}{n_prompt_val or n_gen_val}: "
                        f"{row['avg_ts']:.2f} tok/s (+/- {row['stddev_ts']:.2f})"
                    )

    (run_dir / "stdout.jsonl").write_text("\n".join(log_lines) + "\n")
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
    parser.add_argument(
        "--output", type=Path, default=Path("results.jsonl"),
        help="results file to append rows to (default: ./results.jsonl). Point this at a shared "
        "results.jsonl (e.g. ../results.jsonl) to keep MLX rows alongside llama.cpp ones.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="directory to write per-run raw output under (default: ./runs)",
    )
    args = parser.parse_args()

    with args.models.open("rb") as f:
        models_config = tomllib.load(f)
    with args.sweep.open("rb") as f:
        sweep_config = tomllib.load(f)

    if models_config.get("runtime") != "mlx":
        sys.exit(f"{args.models} declares runtime={models_config.get('runtime')!r}, but mlx_bench.py only drives mlx")

    sweep_name = args.sweep.stem

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    models = select_models(models_config["models"], args.model)
    if not models:
        sys.exit(f"--model {args.model} matched no models in {args.models}")

    total = 0
    for model_cfg in models:
        total += run_model(model_cfg, sweep_config.get("args", {}), sweep_name, args.label, args.output, args.runs_dir)

    print(f"done: {total} total result row(s)")


if __name__ == "__main__":
    main()
