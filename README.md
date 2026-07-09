# local-coding-bench

Benchmarks local LLM inference for coding tasks — speed (tokens/sec across
prompt sizes, context depth, and sustained load) and quality (HumanEval-style
pass rate, including a harder variant and a fast-vs-thinking-mode comparison)
— across model sizes, quantizations, and runtimes (`llama.cpp` and MLX).

Built to answer a specific question: **which model/quant/runtime combination
is actually the best choice for local coding work**, on real hardware,
with both halves of the tradeoff (how fast, how good) measured the same way
so they can be compared directly. Results and writeup: [link to the article
once published].

## Quickstart

    ./setup.sh                                  # llama.cpp + venv + full model roster
    python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label $(hostname -s)
    .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label $(hostname -s)

`setup.sh --models=qwen3-4b,llama-3.1-8b` downloads only those model
families instead of everything (~150GB across all four at both quants, both
formats). `setup.sh --skip-models` skips downloads entirely if you just want
to inspect existing results or build llama.cpp/the venv.

Every script takes `--models` (which model roster — `configs/models.toml`
for llama.cpp, `configs/models-mlx.toml` for MLX) and `--sweep` (what to run
— `configs/sweeps/*.toml`) as separate files, so adding a model doesn't mean
editing every sweep config, and vice versa. `coding_eval.py`'s sweeps
(`coding-eval*.toml`) work with either models file since the args are
runtime-agnostic; `bench.py`/`mlx_bench.py`'s throughput sweeps are
runtime-specific (`standard.toml` vs `standard-mlx.toml`, etc.) since
`llama-bench`'s CLI-flag args and `mlx_bench.py`'s native-type args aren't
the same shape.

## Model roster

Dense and MoE, four size classes, GGUF (`models/gguf/<family>/`) and MLX
(`models/mlx/<family>/<model>-<quant>/`) side by side:

| family | model | size class | architecture |
|---|---|---|---|
| qwen | Qwen3-4B-Instruct-2507 | 4B | dense |
| llama | Meta-Llama-3.1-8B-Instruct | 8B | dense |
| openai | gpt-oss-20b | 21B total / 3.6B active | MoE |
| qwen | Qwen3.6-27B | 27B | dense |

Each model carries its own license from its publisher — check the relevant
Hugging Face repo before using weights beyond this benchmark.

## Layout

- `results.jsonl` — append-only log of throughput results, one JSON object
  per test. Load with `pd.read_json("results.jsonl", lines=True)` or
  `duckdb.sql("select * from 'results.jsonl'")`. Every row has `runtime`
  (`"llama.cpp"` or `"mlx"`), `model_family`, `model_name`, `quant`,
  `n_prompt`/`n_gen`/`n_depth`, and `avg_ts` (tokens/sec) — `llama.cpp` rows
  also carry everything `llama-bench -o jsonl` emits natively (build commit,
  cpu/gpu info, every runtime flag).
- `eval_results.jsonl` / `eval_runs/<run_id>/details.jsonl` — coding-capability
  eval (pass rate): the quality half, alongside `results.jsonl`'s speed half.
- `runs/<run_id>/` — raw stdout/stderr/cmd per throughput invocation, kept
  for provenance. Gitignored locally; safe to delete.
- `configs/models.toml` / `configs/models-mlx.toml` — the model roster
  (family, name, quant, path) for each runtime. This is the one place a new
  model gets added; every sweep picks it up automatically.
- `configs/sweeps/*.toml` — sweep parameters only (no models, no
  runtime/binary — those come from whichever `--models` file you pass):
  - `standard` / `standard-mlx` — pp512/pp1024/tg128 at full GPU offload.
  - `depth` / `depth-mlx` — same, but at KV-cache depths
    0/4096/16384/32768, to see how throughput degrades as context grows.
  - `cpu-baseline` — same models, `-ngl 0`, to measure the GPU speedup (no
    MLX equivalent — MLX's whole premise is unified-memory GPU execution,
    so CPU-only isn't a meaningful comparison point there).
  - `sustained` — long generation (1500 tokens) x3 reps, to check for
    thermal-throttling drift rep-over-rep (coarse signal — see caveat
    below).
  - `coding-eval` — HumanEval-subset pass@1, fast mode. Runtime-agnostic —
    used with either `--models` file.
  - `coding-eval-hard` — same, but HumanEval+'s stricter test suites on the
    harder half of the problem set.
  - `coding-eval-thinking` — reasoning enabled. Run against the full
    roster; it's a documented no-op on models that were never trained with
    a thinking mode (see "Fast mode vs. thinking mode" below), so there's
    no need for a separate thinking-only roster.
- `scripts/bench.py` — driver for `llama.cpp`; shells out to `llama-bench -o
  jsonl` once per model and appends normalized rows.
- `scripts/mlx_bench.py` — driver for MLX; run with `.venv/bin/python`.
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
  edge-case asserts vs. HumanEval's handful) — the harder eval, see below.

## Python environment

`setup.sh` creates one venv at the repo root (`.venv`) covering both
runtimes' Python-side needs: `mlx`, `mlx-lm`, a pinned `transformers`
(`mlx-lm` 0.31.3 declares `transformers>=5.0.0` but is actually broken
against anything past exactly `5.0.0` — this is a real upstream bug, not a
typo), and `numpy` (`mlx-lm` depends on it anyway; `coding_eval.py`'s test
execution needs it too). `llama.cpp` itself needs no venv — it's a compiled
binary under `runtimes/llama.cpp/build/bin/`.

`bench.py` and the llama.cpp path of `coding_eval.py` don't touch `mlx_lm`,
so plain `python3` works for those; `.venv/bin/python` works for everything
and is the simpler thing to default to.

## MLX depth-sweep methodology

`llama-bench` isolates "process N new tokens against an existing KV cache of
depth D" as one measurement. `mlx_lm` has no built-in equivalent, so
`mlx_bench.py` builds one explicitly: for each depth D it constructs a fresh
`prompt_cache`, primes it with D filler tokens (untimed), then measures
against that primed cache — `n_prompt` new tokens for the pp row, `n_gen`
generated tokens for the tg row. This isolation matters: measuring pp as a
single `depth + n_prompt` prefill without cache reuse would conflate
attention-over-cache cost with raw batch throughput, which increases with
batch size and trends the wrong direction (i.e. pp throughput would falsely
*increase* with depth).

## Thermal/sustained-load caveat

`sustained.toml` reruns the same long generation `-r 3` times back-to-back
and relies on `samples_ts` (per-rep throughput) trending downward as a proxy
for thermal throttling. This is a coarse signal — it can't see
within-generation degradation, only rep-over-rep drift over the ~15s–90s
each rep takes. Good enough to flag whether throttling is happening at all,
not a rigorous thermal study.

## Coding-eval methodology

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
  `mlx_bench.py`) additionally checks for `config.json`, at least one
  `*.safetensors` file, and no `*.incomplete` markers. GGUF is a single
  file that `hf download` renames into place atomically on completion, so
  `.exists()` alone is fine there.
- **Thinking/reasoning models can return empty completions under a token
  cap.** Qwen3.x emits hidden `<think>...</think>` reasoning before the
  final answer; gpt-oss's harmony format does the same via an "analysis"
  channel. `llama-server`'s chat-completions API splits this into a
  separate `reasoning_content` field, so if the model doesn't finish
  reasoning within `max_tokens`, `content` comes back empty — that's a
  code-extraction failure, not evidence the model can't code. Two
  different knobs suppress it: `enable_thinking: false` for Qwen3.x,
  `reasoning_effort: "low"` for gpt-oss (it ignores `enable_thinking`
  entirely). Both are set on every request via `chat_template_kwargs`
  (llama-server) / a kwarg to `apply_chat_template` (MLX) — harmless no-ops
  for chat templates that don't reference them.
- **Test execution always runs under a fixed interpreter** (`TEST_PYTHON`,
  `.venv`), never `sys.executable` — otherwise results depend on which
  Python happened to launch the harness, and HumanEval+'s numpy-dependent
  test code silently breaks under a bare interpreter that doesn't have it.

### Fast mode vs. thinking mode

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

## Contributing

Running this on your own hardware and want your numbers included? See
[CONTRIBUTING.md](CONTRIBUTING.md) — `results.jsonl`/`eval_results.jsonl`
are designed to hold multiple machines' runs side by side.

## License

MIT for the code in this repo. See [LICENSE](LICENSE) for what that does
and doesn't cover (model weights, llama.cpp, mlx-lm, and the HumanEval
datasets each carry their own separate licenses).
