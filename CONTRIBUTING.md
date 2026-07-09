# Contributing

## Adding your own hardware's results

`results.jsonl` and `eval_results.jsonl` are append-only and designed to hold
runs from multiple machines side by side — every row is tagged with `label`
(your machine name), plus `cpu_info`/`gpu_info` (llama.cpp rows) or
`gpu_info` (MLX rows) captured automatically. To contribute:

1. `./setup.sh` (see README for options — you don't need every model).
2. Run whichever sweeps apply to the hardware/models you have:

       python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label your-machine-name
       .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label your-machine-name

3. Open a PR with the new rows appended to `results.jsonl` /
   `eval_results.jsonl` (and the corresponding `runs/`/`eval_runs/` detail
   files if you want to include them — they're gitignored locally but worth
   attaching to the PR for provenance).

Please don't edit or reorder existing rows — this is meant to accumulate,
not get rewritten.

## Adding a new model

Add a `[[models]]` block to `configs/models.toml` (and `configs/models-mlx.toml`
if you also have the MLX version) pointing at the new model file — every
sweep picks it up automatically, no need to touch anything under
`configs/sweeps/`.

## Adding a new sweep

Copy a file in `configs/sweeps/`, keep just an `[args]` block, and set it to
whatever you want to sweep. No models or runtime info belongs in a sweep
file — those come from whichever `--models` file it's run with.

## Adding a new runtime

Follow `scripts/mlx_bench.py`'s precedent: a new script that takes
`--models`/`--sweep` the same way, drives the runtime however it needs to,
and appends rows to `results.jsonl` with the same field names (`runtime`,
`n_prompt`, `n_gen`, `n_depth`, `avg_ts`, `model_family`, `model_name`,
`quant`, ...). Add a matching `configs/models-<runtime>.toml`.
