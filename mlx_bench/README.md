# mlx_bench

An MLX throughput benchmark. `mlx_lm` ships no built-in benchmark tool the
way `llama.cpp` ships `llama-bench`, so `bench.py` here hand-builds one,
measuring the same pp/tg/depth shape a `llama-bench`-driven benchmark would.

`bench.py` is fully self-contained: its own model roster, its own sweep
configs, and every path it touches — models, output file, run logs — is
either passed in as an argument or resolved relative to wherever you invoke
it from. It doesn't assume it's running inside a larger project or reach
outside its own arguments for anything. Its output row schema is
deliberately kept compatible with a `llama-bench`-style driver's
(`runtime`, `model_family`, `model_name`, `quant`, `n_prompt`/`n_gen`/`n_depth`,
`avg_ts`/`stddev_ts`, ...) so that *if* you point `--output` at a file
another such driver also writes to, rows from both stay directly
comparable — that's opt-in via `--output`, not assumed.

## Concepts

Every row this produces measures one of two things, at a given quant and a
given depth of prior context:

- **pp** (prompt processing / prefill) — throughput processing brand-new
  prompt tokens, before the model starts responding.
- **tg** (generation) — throughput producing tokens one at a time, after
  that.
- **depth** — how many tokens are already sitting in the model's key-value
  cache (its running memory of prior context, e.g. earlier conversation
  turns) before the timed measurement starts. `0` means an empty context;
  larger values simulate a conversation or agentic session that's already
  gone on a while. Throughput drops as depth grows — the whole reason this
  script sweeps over multiple depths instead of only measuring at `0`.

Both are reported in tokens/sec (`avg_ts`/`stddev_ts` in the output).

## Usage

Run from the repo root — `--output`/`--runs-dir` default to
`./results.jsonl`/`./runs`, so they land at the repo root alongside any
other driver's output there, for direct comparison:

    .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml [--label m5-max-macbook]
    .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml --model qwen3-coder-next

Or run from within this directory for a fully standalone result, with
`results.jsonl`/`runs/` created right here instead:

    cd mlx_bench && .venv/bin/python bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml

`--models`/`--sweep`/`--output`/`--runs-dir`, and every model's `path` in
the roster TOML, are all resolved relative to wherever you run the command
from — pick whichever of the two invocation styles above suits you and keep
paths consistent with it.

Uses the project's `.venv` (`mlx`, `mlx-lm`, `transformers`, `numpy`).

## Layout

- `bench.py` — the driver script; see below for how it works.
- `configs/models.toml` — MLX model roster (`family`/`name`/`quant`/`path`,
  plus `hf_repo` for downloading).
- `configs/sweeps/standard.toml`, `configs/sweeps/depth.toml` — MLX-specific
  sweep args (native Python types, not `llama-bench` CLI flags — see [Split
  models / sweeps / runtime](#why-a-separate-roster-and-sweep-files) below).

### Why a separate roster and sweep files

A sweep's `[args]` shape is runtime-specific — a CLI-flag-based benchmark
takes flags like `-p`/`-d`/`-ngl`, `bench.py` here takes native Python-typed
args (`p`, `d`, lists and ints straight out of TOML) — so sweep files exist
once per runtime rather than being shared. The model roster is per-runtime
for a more basic reason: an MLX model and, say, a GGUF build of the same
model are different files on disk entirely, so `path` has to differ even
when `family`/`name`/`quant` line up.

## The core problem: no built-in "measure against a primed cache"

A CLI benchmark like `llama-bench` can be told, in one invocation, "process
N new prompt tokens against a KV cache that already holds D tokens" — a
`-d` flag. `mlx_lm` has a `generate`/`stream_generate` API and a
`prompt_cache` object, but nothing that measures throughput at a given
cache depth out of the box. `bench.py`'s job is entirely this: reconstruct
that one measurement using the pieces `mlx_lm` does provide.

The approach (`primed_cache()` + `run_point()`):

1. Build a fresh `prompt_cache` via `make_prompt_cache(model)`.
2. If depth `D > 0`, run an untimed generation of `D` filler tokens into
   that cache — this is the "already has D tokens of context" setup step,
   deliberately not timed.
3. Measure against the now-primed cache:
   - **pp row**: feed `n_prompt` new filler tokens, `max_tokens=1`. The
     `stream_generate` response's `prompt_tps` field is the throughput for
     ingesting those `n_prompt` tokens specifically — not the D tokens that
     primed the cache.
   - **tg row**: feed a single token, `max_tokens=n_gen`. The response's
     `generation_tps` is throughput generating those `n_gen` tokens from the
     D-token cache.

Both rows use a *fresh* primed cache (`primed_cache()` is called again for
the tg measurement, not reused from the pp measurement) so the two numbers
don't contaminate each other.

### Why the cache has to be primed and reused, not just requested as one long prompt

The tempting shortcut is a single call: send `D + n_prompt` tokens as one
prompt and read `prompt_tps` off that. This is wrong, not just less
elegant. `prompt_tps` from a single large prefill mixes two different
costs — attention over the existing cache (which should scale with `D`) and
raw prefill batch throughput (which *improves* with a bigger batch,
independent of `D`). At large `D` those effects point in opposite
directions, so a naive combined measurement can show pp throughput
increasing with depth — the opposite of the real effect this benchmark
exists to catch: prompt processing and generation both actually get slower
as the context already in the cache grows, which matters directly for any
long-running conversation or agentic session. Priming the cache with an
untimed step and then measuring a small, fixed-size `n_prompt` against it
isolates the depth-dependent cost from the batch-size-dependent one.

## Warmup: two layers, for two different costs

- **Whole-model warmup**, once per model load, before the sweep starts
  (`timed_generate(..., filler_tokens(tokenizer, 16), max_tokens=8)`).
  Covers first-load costs — weight materialization, MLX's lazy evaluation
  kicking in for the first time.
- **Per-point warmup**, inside `run_point()`: each `(depth, n_prompt)` or
  `(depth, n_gen)` combination is measured `reps + 1` times, and the first
  sample is thrown away (`if i > 0: ...append(...)`). MLX can pay a
  one-time cost — graph construction, buffer allocation — the *first* time
  it sees a given prompt/generation shape specifically, not just the first
  time it runs anything. A shape seen for the first time mid-sweep (a new
  `n_prompt` value, a new depth) would otherwise skew a small-rep average
  upward.

`reps` defaults to 5 (`args.get("r", 5)`) unless a sweep file overrides it
— see `configs/sweeps/*.toml` in this directory.

## Filler tokens, generated once and cached

`filler_tokens()` tokenizes a fixed block of repeated text
(`"The quick brown fox..." * 4000`) once per process, into a module-level
`_filler_ids_cache`, and slices/repeats that list to whatever length a
given call needs. Re-tokenizing per call would add tokenizer overhead to
every single measurement point in the sweep — the filler content itself
doesn't matter (this is a throughput benchmark, not a quality one), only
its length, so caching the token IDs once and slicing is free
correctness-wise and removes that overhead entirely.

## Model-readiness check is stricter than `Path.exists()`

`hf download --local-dir` creates the destination directory immediately,
before any file lands in it — so a bare `.exists()` check on an MLX model
directory would call a still-downloading model "ready" and crash partway
through a sweep on missing `.safetensors` files, potentially after already
successfully benchmarking several other models in the same run.
`model_ready()` instead checks: is it a directory, does `config.json`
exist, is there at least one `*.safetensors` file, and are there zero
`*.incomplete` markers anywhere under it.

## Output schema is designed for cross-runtime comparability

Every row this appends uses a deliberately generic field set — `runtime`,
`model_family`, `model_name`, `quant`, `n_prompt`, `n_gen`, `n_depth`,
`avg_ts`, `stddev_ts`, `run_id`, `config`, `label` — chosen so that rows
from a completely different benchmark driver (a different runtime, a
different script, run separately) can sit in the same file and be grouped/
compared without special-casing. `runtime: "mlx"` and an MLX-only
`mlx_lm_version` field are the only things that distinguish one of these
rows from, say, a `llama-bench`-driven one. Whether that actually happens
depends entirely on where you point `--output` — see Usage above.

Raw per-invocation output (every row this run produced, not just the
aggregated numbers) is also written to `<runs-dir>/<run_id>/stdout.jsonl`,
for provenance.

## What it doesn't do

- **No CPU-only mode.** There's no `-ngl 0` equivalent for MLX in this
  script, because there's nothing to offload — MLX's whole premise is
  unified-memory GPU execution, so a CPU-only comparison point doesn't
  exist the way it might for a CPU-capable runtime.
- **One model loaded per script invocation.** The sweep loop (`for depth
  ... for p ...`) runs entirely against one already-loaded `model` +
  `tokenizer`, reloaded fresh only when `run_model()` is called again for
  the next model in the roster. Cheap within a model's sweep, but there's
  no cross-model batching or shared-memory tricks — each model pays its own
  full load cost.
