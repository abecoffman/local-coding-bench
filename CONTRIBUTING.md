# Contributing

## Running tests

    python3 -m unittest discover -s tests -v

Stdlib `unittest`, no extra dependencies — covers the pure-logic pieces
that are cheapest to break silently: `scripts/_common.py`'s `select_models`/
`slugify`/`find_repo_root`, `aggregate.py`'s test-point grouping,
`coding_eval.py`'s `extract_code`, and that `configs/models.toml` and
`mlx_bench/configs/models.toml` list the same models (nothing else enforces
that when either file is hand-edited). Doesn't cover the parts that need a
real model loaded or a real `llama-server` running — those are exercised by
actually running a sweep, not by this suite.

## Running the benchmarks on your own hardware

`results.jsonl`, `eval_results.jsonl`, and `results-aggregated.{jsonl,csv}`
are gitignored, not committed — running the benchmarks produces them locally
for you to inspect (`pd.read_json(...)`, `duckdb.sql(...)`, or just open the
CSV), but there's no shared file to append to or open a PR against. Every
row is still tagged with `label` (a machine name you choose via `--label`)
plus `cpu_info`/`gpu_info`, in case you're comparing multiple runs or
machines locally.

    ./setup.sh                                  # see README for options — you don't need every model
    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label your-machine-name
    .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml --label your-machine-name
    .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label your-machine-name
    python3 scripts/aggregate.py results.jsonl -o results-aggregated.jsonl --csv results-aggregated.csv

Want to share what you found? The raw/aggregated files above stay local,
but a written write-up is small and worth contributing: add
`analysis/<your-machine>.md` (see [`analysis/m5-max-macbook.md`](analysis/m5-max-macbook.md)
for the shape one takes) and open a PR with just that file.

## Adding a new model

Add a `[[models]]` block to `configs/models.toml` (and
`mlx_bench/configs/models.toml` if you also have the MLX version) with the
new model's `family`/`name`/`quant`/`path`, plus `hf_repo` (and `hf_files`
for the GGUF roster — one or more filenames within that repo; a list even
for a single file) so `scripts/download_models.py` can fetch it:

    [[models]]
    family = "qwen"
    name = "Some-New-Model"
    quant = "Q4_K_M"
    path = "models/gguf/qwen/Some-New-Model-Q4_K_M.gguf"
    hf_repo = "unsloth/Some-New-Model-GGUF"
    hf_files = ["Some-New-Model-Q4_K_M.gguf"]

That's the only file that needs editing — every sweep and `setup.sh`
picks the new model up automatically, no need to touch anything under
`configs/sweeps/` or `setup.sh` itself.

## Adding a new sweep

Copy a file in `configs/sweeps/`, keep just an `[args]` block, and set it to
whatever you want to sweep. No models or runtime info belongs in a sweep
file — those come from whichever `--models` file it's run with.

## Adding a new runtime

Follow `mlx_bench/`'s precedent: its own directory, with a `bench.py` that
takes `--models`/`--sweep` the same way plus `--output`/`--runs-dir` (default
`./results.jsonl`/`./runs`, so running from the repo root lines up with
`scripts/bench.py`'s output by default, without either script needing to
know the other exists), and appends rows using the same field names
(`runtime`, `n_prompt`, `n_gen`, `n_depth`, `avg_ts`, `model_family`,
`model_name`, `quant`, ...). Give it its own `configs/models.toml` inside
that directory, with an `hf_repo` per model (and `hf_files` too, if it
downloads individual files the way the GGUF roster does —
`mlx_bench/configs/models.toml` downloads a whole repo per model instead,
since that's how MLX weights are packaged) so `scripts/download_models.py`
can drive its downloads without changes. Add its own README explaining the
runtime-specific pieces (see `mlx_bench/README.md`).
