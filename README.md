# local-coding-bench

## Intro

Benchmarks local LLM inference for coding tasks — speed (tokens/sec across
prompt sizes, context depth, and sustained load) and quality (HumanEval-style
pass rate, including a harder variant and a fast-vs-thinking-mode comparison)
— across model sizes, quantizations, and runtimes (`llama.cpp` and MLX).

Built to answer a specific question: **which model/quant/runtime combination
is actually the best choice for local coding work**, on real hardware, with
both halves of the tradeoff (how fast, how good) measured the same way so
they can be compared directly.

**Requirements**: Apple Silicon macOS only — `llama.cpp`'s numbers here rely
on Metal, and MLX doesn't run anywhere else. Also needs Python 3.11+ (every
driver script uses the stdlib `tomllib` module, added in 3.11) — `setup.sh`
checks for this before doing anything else.

This README walks through, in order: what varies about the models tested,
what varies about the software that runs them, and how this repo actually
measures speed and quality. Actual results — numbers, not just methodology
— live separately, in a machine-specific write-up: see [Example
Analysis](#example-analysis).

## Model Concepts & Models Used

### Concepts

Three things vary about how a model is built or compressed, on top of raw
parameter count (4B–27B — see [Models used](#models-used) below):

#### Quant(ization)

Shrinking a model's weights to fewer bits per parameter (e.g. 4-bit instead
of 16-bit), trading some output quality for a smaller file and faster
inference. This benchmark tests two levels per model: lighter (`Q4_K_M` /
`4bit`) and heavier (`Q8_0` / `8bit`) — the naming differs because
llama.cpp and MLX use different conventions for the same idea.

#### Dense vs. MoE (mixture-of-experts)

A dense model uses every parameter on every token. An MoE model routes each
token through only a subset ("experts") of a larger total parameter count —
why gpt-oss-20b's `params` and `active` columns show two different numbers
(21B and 3.6B) instead of one.

#### Attention variant

How a model implements attention, the mechanism that lets each new token
look back at prior ones through a cache of their key/value representations
instead of reprocessing them (that cache is what [context
depth](#context-depth) refers to). How the cache is organized differs by
model, and that difference is a big part of why these models don't pay the
same [tax for context depth](analysis/m5-max-macbook.md):

- **Grouped-query attention (GQA)** — the common default, and every model
  here uses it in at least some of its layers. Several "query" heads share
  one cached key/value pair instead of every head keeping its own,
  shrinking the cache. Qwen3-4B, Llama-3.1-8B, and Devstral-Small-2-24B all
  pair 32 query heads with 8 shared key/value heads; gpt-oss-20b pairs 64
  query heads with 8 (a wider share); Qwen3.6-27B's and Qwen3-Coder-Next's
  non-linear attention layers pair 24/4 and 16/2 query-to-key/value heads
  respectively.
- **Sliding-window attention** — caps how far back a layer looks,
  regardless of how much context has piled up. gpt-oss-20b alternates a
  128-token sliding window with full-context layers, so half its layers
  never pay for depth beyond the last 128 tokens.
- **Linear/hybrid attention** — a newer design (e.g. Gated DeltaNet) that
  skips the growing cache entirely, keeping a fixed-size running state
  instead. Qwen3.6-27B and Qwen3-Coder-Next both use this for three out of
  every four layers (Qwen3-Coder-Next's config literally calls it
  `full_attention_interval: 4`), falling back to GQA (with a normal growing
  cache) on the fourth. That's the main reason both are the low-tax outliers
  in [Context depth
  tax](analysis/m5-max-macbook.md) — sharing the design
  means sharing the effect.

### Models used

Dense and MoE, a range of sizes, GGUF (`models/gguf/<family>/`) and MLX
(`models/mlx/<family>/<model>-<quant>/`) side by side:

| family | model | params | active | architecture |
|---|---|---|---|---|
| qwen | Qwen3-4B-Instruct-2507 | 4B | 4B | dense |
| llama | Meta-Llama-3.1-8B-Instruct | 8B | 8B | dense |
| openai | gpt-oss-20b | 21B | 3.6B | MoE |
| qwen | Qwen3.6-27B | 27B | 27B | dense |
| mistral | Devstral-Small-2-24B-Instruct-2512 | 24B | 24B | dense |
| qwen | Qwen3-Coder-Next | 80B | 3B | MoE |

Each model carries its own license from its publisher — check the relevant
Hugging Face repo before using weights beyond this benchmark.

Qwen3-Coder-Next and Devstral-Small-2-24B-Instruct-2512 are the newest
additions — both coding-specialized rather than general-purpose instruct
models, filling a gap the original four didn't cover.

## Runtime Concepts & Runtimes Used

### Runtime

The software that actually runs a model — loads the weights, executes the
forward pass, manages the key/value cache. Two are compared in this repo,
and they differ in more than file format.

### Runtimes used

#### llama.cpp

A widely-used C++ inference engine. Runs `GGUF`-format model files,
Metal-accelerated on Apple Silicon. Ships its own CLI benchmark tool
(`llama-bench`) and can run CPU-only (`-ngl 0`) as a baseline — see [GPU vs
CPU](analysis/m5-max-macbook.md).

#### MLX

Apple's own array/ML framework, built for Apple Silicon's unified memory.
`mlx-lm` is the piece that runs LLMs on top of it. No CPU-only mode, since
unified-memory GPU execution is its whole premise, and no built-in
benchmark tool — this repo's [`mlx_bench/`](mlx_bench/README.md) reimplements
one (see [MLX depth-sweep methodology](#mlx-depth-sweep-methodology)).

#### Python environment

`llama.cpp` itself needs no venv — it's a compiled binary under
`runtimes/llama.cpp/build/bin/`. Everything Python-side, for both runtimes,
uses one venv that `setup.sh` creates at the repo root (`.venv`):

- `mlx` / `mlx-lm` — run inference on the MLX side.
- `transformers`, pinned to exactly `5.0.0` — `mlx-lm` 0.31.3 declares
  `transformers>=5.0.0`, but is actually broken against anything past that
  exact version. Real upstream bug, not a typo here.
- `numpy` — `mlx-lm` depends on it anyway, and `coding_eval.py`'s test
  execution needs it too.

Which interpreter to use:

- `bench.py`, and the llama.cpp path of `coding_eval.py`, never touch
  `mlx_lm`, so plain `python3` works for those.
- `.venv/bin/python` works for everything, so it's the simpler default.

## Benchmarking Concepts & Libraries Used

### Concepts

This benchmark measures two axes on every model/quant/runtime combination:
speed and quality. The terms below are what the tooling in [Libraries
used](#libraries-used) reports for each.

#### pp / tg

Prompt processing (ingesting the prompt) vs. token generation (producing
the reply) — the two things `llama-bench` and `mlx_bench/bench.py` time.

#### Prompt size

How many fresh tokens get ingested in one prompt processing measurement
(e.g. `pp512` = 512 tokens). Tested at a couple of fixed sizes to see
whether throughput holds steady as the prompt itself gets longer.

#### Context depth

Tokens already held in the model's key-value cache (its running memory of
prior context, e.g. from earlier conversation turns) before a new prompt is
processed, as opposed to starting from an empty context. `n_depth` in the
raw data. See [Context depth tax](analysis/m5-max-macbook.md).

#### Sustained load

Many long generations run back-to-back rather than a single one-off run, to
check for rep-over-rep slowdown (e.g. thermal throttling) instead of a
snapshot. See [Thermal/sustained-load
caveat](#thermalsustained-load-caveat).

#### tok/s (tokens/sec)

The throughput unit for both pp and tg. Higher is faster.

#### pass@1

The fraction of coding problems solved correctly on the first (and only)
generated attempt — `coding_eval.py`'s quality metric.

#### Thinking mode

Some models can emit an internal chain of reasoning — working through the
problem step by step — before writing their actual answer, when a
`thinking`/`reasoning` flag is turned on in the request. That reasoning is
usually hidden from the final reply but still costs generation time, since
the model has to produce and pay for every reasoning token before it starts
the answer proper. Off by default in this benchmark's coding-eval sweeps
("fast mode"): on in one sweep specifically to measure the tradeoff. Not
every model supports it — some releases (e.g. `Qwen3-4B-Instruct-2507`,
`Devstral-Small-2-24B-Instruct-2512`) were never trained with a distinct
reasoning mode, so turning the flag on for them changes little to nothing.
See [Fast mode vs. thinking mode](#fast-mode-vs-thinking-mode) for exactly
how this repo toggles it, and [Thinking
mode](analysis/m5-max-macbook.md) under Quality findings for
what turning it on actually cost/bought per model tested.

### Libraries used

Speed numbers come from one tool per runtime: llama.cpp's own `llama-bench`
(driven by `scripts/bench.py`), and this repo's [`mlx_bench/`](mlx_bench/README.md)
directory, which reimplements equivalent measurements for MLX since it has
none built in. Quality numbers come from a third, runtime-agnostic script,
`scripts/coding_eval.py`.

#### Speed benchmark methodology

Every speed row measures one of two things:

- **pp** (prompt processing / prefill) — throughput processing `n_prompt`
  brand-new tokens. How fast the model chews through a prompt before it
  starts responding.
- **tg** (generation) — throughput producing `n_gen` tokens one at a time,
  after that.

Two more fields show up on every row:

- `n_depth` — tokens already in the KV cache before timing starts (e.g.
  from a prior conversation turn). `0` means an empty context.
- `n_gpu_layers` (`-ngl`) — model layers offloaded to the GPU. `99` is full
  Metal offload, `0` is CPU-only. llama.cpp only — MLX has no CPU-only mode
  (see [GPU vs CPU](analysis/m5-max-macbook.md)).

**Tooling differs by runtime**, since only one ships its own benchmark
harness:

- **llama.cpp** — `bench.py` shells out to the `llama-bench` binary
  directly. It warms up before timing (on by default) and repeats each
  point `-r` times (5 by default, 2–3 in some sweeps — see
  [Layout](#layout)), reporting mean/stddev across those repeats natively.
- **MLX** — no built-in harness, so `mlx_bench/bench.py` reimplements the
  same measurement (details: [MLX depth-sweep
  methodology](#mlx-depth-sweep-methodology)). It runs `reps + 1` times per
  point and drops the first sample, since MLX pays a one-time compilation
  cost the first time it sees a given prompt/generation shape. There's also
  a one-time whole-model warmup right after load, before any sweep point
  runs.

**A wrinkle in `results-aggregated.csv`**: its `n_runs` column is not the
`-r`/`reps` count above. `results.jsonl` is append-only across however many
separate script invocations you've run — different sessions, reruns,
contributed machines — and each row is already one invocation's mean across
its own repeats. `n_runs=6` means six *separate invocations* landed on the
same test point, each already averaging 2–5 internal repeats.

#### MLX depth-sweep methodology

`llama-bench` isolates "process N new tokens against an existing KV cache of
depth D" as one measurement. `mlx_lm` has no built-in equivalent, so
`mlx_bench/bench.py` builds one explicitly: for each depth D it constructs a fresh
`prompt_cache`, primes it with D filler tokens (untimed), then measures
against that primed cache — `n_prompt` new tokens for the pp row, `n_gen`
generated tokens for the tg row.

This isolation matters. Measuring pp as a single `depth + n_prompt` prefill
without cache reuse would conflate attention-over-cache cost with raw batch
throughput, which increases with batch size — trending the wrong direction
(pp throughput would falsely *increase* with depth).

See [`mlx_bench/README.md`](mlx_bench/README.md) for the rest of its
design — warmup handling, the pp/tg measurement trick, model-readiness
checks, and how its output stays schema-compatible with `bench.py`'s.

#### Thermal/sustained-load caveat

`sustained.toml` reruns the same long generation `-r 3` times back-to-back
and relies on `samples_ts` (per-rep throughput) trending downward as a proxy
for thermal throttling. This is a coarse signal — it can't see
within-generation degradation, only rep-over-rep drift over the ~15s–90s
each rep takes. Good enough to flag whether throttling is happening at all,
not a rigorous thermal study.

#### Coding-eval methodology

`coding_eval.py` runs a subset (`n_problems`, default 20 of 164) of a
HumanEval-family dataset per model/quant/runtime: greedy decoding (temp 0),
single sample, pass@1, code extracted from the last fenced ` ```python `
block in the response and executed against the problem's own `check()`
assertions in a subprocess with a timeout, no further sandboxing (acceptable
locally, not for untrusted code). It's a subset for time/practicality
reasons, not full-benchmark rigor — prefer relative comparisons (Q4 vs Q8
pass rate for the *same* model) over cross-model bragging rights unless you
bump `n_problems` up.

Known gotchas, all handled but worth knowing if you extend this:

- **MLX model readiness** needs more than `Path.exists()`. `hf download
  --local-dir` creates the destination directory immediately, before any
  file lands, so a still-downloading MLX model looks "ready" to a bare
  existence check. `model_ready()` (in both `coding_eval.py` and
  `mlx_bench/bench.py`) additionally checks for `config.json`, at least one
  `*.safetensors` file, and no `*.incomplete` markers. GGUF is a single
  file that `hf download` renames into place atomically on completion, so
  `.exists()` alone is fine there.
- **Thinking/reasoning models can return empty completions under a token
  cap.** Qwen3.x emits hidden `<think>...</think>` reasoning before the
  final answer; gpt-oss's harmony format does the same via an "analysis"
  channel. `llama-server`'s chat-completions API splits this into a
  separate `reasoning_content` field — so if the model doesn't finish
  reasoning within `max_tokens`, `content` comes back empty. That's a
  code-extraction failure, not evidence the model can't code.

  Two different knobs suppress it: `enable_thinking: false` for Qwen3.x,
  `reasoning_effort: "low"` for gpt-oss (it ignores `enable_thinking`
  entirely). Both are set on every request — harmless no-ops for chat
  templates that don't reference them.
- **Test execution always runs under a fixed interpreter** (`TEST_PYTHON`,
  `.venv`), never `sys.executable` — otherwise results depend on which
  Python happened to launch the harness, and HumanEval+'s numpy-dependent
  test code silently breaks under a bare interpreter that doesn't have it.

##### Fast mode vs. thinking mode

Disabling reasoning by default is a deliberate methodology choice: it
evaluates models in "fast/interactive local coding assistant" mode, the
realistic mode for latency-sensitive local usage, not their best-possible
reasoning mode. Three config knobs (`[args]` in a coding-eval config)
control this axis:

- `dataset = "humanevalplus"` — far stricter test suites than base
  HumanEval, so shallow or lucky solutions get caught instead of
  rubber-stamped. Worth using by default: base HumanEval saturates at 100%
  for every model except Llama-3.1-8B in fast mode, leaving no room to show
  a quality gap between quants or a reasoning gap between modes.
- `problem_offset = 144` — HumanEval gets harder toward the end by
  community consensus; the last 20 problems discriminate better than the
  default first-20.
- `thinking = true` — re-enables reasoning (`enable_thinking: true`,
  `reasoning_effort: "high"`) and needs a much larger `max_tokens` (4096+)
  since reasoning alone can run to thousands of tokens. This is a no-op on
  models never trained with a thinking mode — e.g. `Qwen3-4B-Instruct-2507`
  is an explicitly non-thinking release (Alibaba ships separate
  Instruct/Thinking variants) and shows near-identical gen times with
  `thinking` on vs. off, whereas Qwen3.6-27B goes from ~2s/problem (fast)
  to ~130s/problem (thinking) — reasoning genuinely engaging, not just
  declared.

### Quickstart

    ./setup.sh                                  # llama.cpp + venv + full model roster
    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label $(hostname -s)
    .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label $(hostname -s)

`setup.sh --models=qwen3-4b,llama-3.1-8b` downloads only those model
families instead of everything (~500GB across all six at both quants, both
formats — Qwen3-Coder-Next alone is roughly half of that, since even its
3B-active MoE weights still ship at their full 80B-total size). `setup.sh
--skip-models` skips downloads entirely if you just want to inspect
existing results or build llama.cpp/the venv.

Every script takes `--models` (which model roster — `configs/models.toml`
for llama.cpp, `mlx_bench/configs/models.toml` for MLX) and `--sweep` (what
to run — `configs/sweeps/*.toml`, or `mlx_bench/configs/sweeps/*.toml` for
MLX) as separate files, so adding a model doesn't mean editing every sweep
config, and vice versa. `coding_eval.py`'s sweeps (`coding-eval*.toml`,
under the top-level `configs/sweeps/`) work with either models file since
the args are runtime-agnostic; `bench.py`/`mlx_bench/bench.py`'s throughput
sweeps are runtime-specific (`configs/sweeps/standard.toml` vs.
`mlx_bench/configs/sweeps/standard.toml`, etc.) since `llama-bench`'s
CLI-flag args and `mlx_bench/bench.py`'s native-type args aren't the same
shape — see [`mlx_bench/README.md`](mlx_bench/README.md) for that
directory's own layout.

All three scripts also take a repeatable `--model` filter — a
case-insensitive substring match against `"<family> <name> <quant>"`, OR'd
together across repeats — so re-benchmarking one model after adding it to
the roster doesn't mean re-running the other five:

    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --model qwen3-coder-next --label $(hostname -s)

Omit `--model` to run the whole roster, as in the Quickstart above.

### Layout

- `results.jsonl` — append-only log of throughput results, one JSON object
  per test, generated locally the first time you run a speed sweep.
  Gitignored, not committed — this repo's own numbers are baked into the
  tables below as static results, not read live from this file. Load your
  own copy with `pd.read_json("results.jsonl", lines=True)` or
  `duckdb.sql("select * from 'results.jsonl'")`. Every row has `runtime`
  (`"llama.cpp"` or `"mlx"`), `model_family`, `model_name`, `quant`,
  `n_prompt`/`n_gen`/`n_depth`, `n_gpu_layers`, and `avg_ts`/`stddev_ts`
  (tokens/sec).

  `llama-bench -o jsonl` natively emits ~50 fields per row. `bench.py` keeps
  only the ones that vary or are meant to — the ~30 that are always fixed
  CLI defaults (thread count, batch size, KV cache dtype, ...) aren't
  repeated on every row. Full unfiltered output is still in
  `runs/<run_id>/stdout.jsonl` if you need it.
- `results-aggregated.jsonl` / `.csv` — derived from your local
  `results.jsonl` by `scripts/aggregate.py` (regenerate with
  `python3 scripts/aggregate.py results.jsonl -o results-aggregated.jsonl --csv results-aggregated.csv`).
  Also gitignored. Groups repeated runs of the same (model, quant, runtime,
  hardware-mode, test-shape) into mean/stddev/n — the `.csv` is the flat
  table meant for skimming, the `.jsonl` keeps the full per-run sample list.
- `eval_results.jsonl` / `eval_runs/<run_id>/details.jsonl` — coding-capability
  eval (pass rate): the quality half, alongside `results.jsonl`'s speed half.
  Also local/gitignored.
- `runs/<run_id>/` — raw stdout/stderr/cmd per throughput invocation, kept
  for provenance. Gitignored locally; safe to delete.
- `configs/models.toml` — the llama.cpp model roster (family, name, quant,
  path, plus `hf_repo`/`hf_files` for `scripts/download_models.py`). This is
  the one place a new llama.cpp model gets added; every llama.cpp sweep
  *and* `setup.sh` pick it up automatically — nothing else needs editing.
  (MLX has its own roster — see `mlx_bench/` below.)
- `configs/sweeps/*.toml` — sweep parameters only (no models, no
  runtime/binary — those come from whichever `--models` file you pass):
  - `standard` — pp512/pp1024/tg128 at full GPU offload.
  - `depth` — same, but at KV-cache depths 0/4096/16384/32768, to see how
    throughput degrades as context grows.
  - `cpu-baseline` — same models, `-ngl 0`, to measure the GPU speedup (no
    MLX equivalent — MLX's whole premise is unified-memory GPU execution,
    so CPU-only isn't a meaningful comparison point there).
  - `sustained` — long generation (1500 tokens) x3 reps, to check for
    thermal-throttling drift rep-over-rep (coarse signal — see caveat
    below).
  - `coding-eval` — HumanEval-subset pass@1, fast mode. Runtime-agnostic —
    used with either `--models` file, llama.cpp's or MLX's.
  - `coding-eval-hard` — same, but HumanEval+'s stricter test suites on the
    harder half of the problem set.
  - `coding-eval-thinking` — reasoning enabled. Run against the full
    roster; it's a documented no-op on models that were never trained with
    a thinking mode (see "Fast mode vs. thinking mode" above), so there's
    no need for a separate thinking-only roster.
- `mlx_bench/` — the MLX side of the speed benchmark: its own `bench.py`,
  `configs/models.toml`, and `configs/sweeps/{standard,depth}.toml`,
  mirroring the llama.cpp files above but MLX-specific. Self-contained with
  its own [README](mlx_bench/README.md) covering its design; still writes
  into the same top-level `results.jsonl` as `scripts/bench.py`, so both
  runtimes' rows stay directly comparable.
- `scripts/bench.py` — driver for `llama.cpp`; shells out to `llama-bench -o
  jsonl` once per model and appends normalized rows.
- `scripts/download_models.py` — downloads a roster via `hf download`,
  driven entirely by each model's `hf_repo`/`hf_files` in `configs/models.toml`
  / `mlx_bench/configs/models.toml`. This is what `setup.sh` calls; supports
  the same `--model` filter as the benchmark scripts, plus `--dry-run` to
  preview commands without downloading anything.
- `scripts/coding_eval.py` — HumanEval-family pass@1 eval, runs against
  either runtime. Run with `.venv/bin/python` regardless of which runtime
  you're evaluating — it's the only Python that has all the pieces
  (`mlx_lm` for the MLX path, `numpy` for test execution regardless of
  path).
- `data/humaneval.jsonl` — 164 HumanEval problems (OpenAI, MIT), converted
  from the `openai/openai_humaneval` parquet dataset for plain stdlib
  `json` loading.
- `data/humanevalplus.jsonl` — same 164 problems with EvalPlus's much
  larger generated test suites per problem (Apache-2.0; thousands of
  edge-case asserts vs. HumanEval's handful) — the harder eval, see above.
  Copied verbatim from `evalplus/humanevalplus`'s own `test.jsonl` — it
  already ships as JSONL upstream, nothing to convert.
- `scripts/fetch_data.py` — regenerates both files above from their
  upstream sources (needs `pip install pyarrow`, not part of the main
  `setup.sh` install). Not something you run day to day; it's what makes
  the two files above reproducible rather than opaque committed blobs.
- `tests/` — `unittest` coverage for the pure-logic pieces (`scripts/_common.py`,
  `aggregate.py`'s grouping, `coding_eval.py`'s code extraction). Run with
  `python3 -m unittest discover -s tests -v`; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Example Analysis

The actual speed and quality numbers this benchmark produces — the whole
reason it exists — live outside this README, in a machine-specific
analysis doc: [`analysis/m5-max-macbook.md`](analysis/m5-max-macbook.md).

Numbers depend on the exact hardware, OS/driver versions, and thermal
conditions at measurement time, so this repo treats them as one worked
example rather than baking them into the main documentation as if they
were universal. Run the benchmarks yourself (see
[Quickstart](#quickstart)) to get your own local
`results.jsonl`/`eval_results.jsonl` (gitignored, not committed — see
[Layout](#layout)), and write up your own `analysis/<your-machine>.md`
from them the same way — the existing doc is a template for the shape
that write-up can take.

## Contributing

Want to add a model, sweep, or runtime, or run the benchmarks on your own
hardware? See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT for the code in this repo. See [LICENSE](LICENSE) for what that does
and doesn't cover (model weights, llama.cpp, mlx-lm, and the HumanEval
datasets each carry their own separate licenses).
