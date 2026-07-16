# Benchmark analysis — M5 Max MacBook Pro

Full results and write-up from one specific machine, produced by running
the benchmarks in this repo (see the project README for what's actually
being measured and why, and its Quickstart section to reproduce this
yourself). Numbers below are one example run, not a universal verdict:
actual throughput and quality depend on the exact hardware, OS/driver
versions, and thermal conditions at measurement time. This is the example
analysis doc this repo ships with; running the benchmarks on your own
hardware produces your own local `results.jsonl`/`eval_results.jsonl`
(gitignored, not committed) that you can write up the same way, as
`analysis/<your-machine>.md`.

**Hardware**: MacBook Pro, Apple **M5 Max** (18 CPU cores: 6 Super + 12
Performance; 40 GPU cores), **128GB** unified memory, macOS 26.5.2.

## Speed at a glance

Tables below are hand-derived from this machine's `results-aggregated.csv`;
the full per-run data (stddev, sample counts) lives there and in
`results-aggregated.jsonl`, raw per-invocation rows in `results.jsonl`. Row
order matches the README's model roster throughout. See the README's Speed
benchmark methodology section for what pp/tg mean and exactly how each
number was measured.

### Generation speed

Tokens/sec while generating (`tg128`) — the number that matters most for an
interactive coding session, since it's how fast you can read the model's
output:

| model | params | active | Q4 llama.cpp | Q4 MLX | Q8 llama.cpp | Q8 MLX |
|---|---|---|---|---|---|---|
| Qwen3 | 4B | 4B | 152 | 184 | 104 | 111 |
| Llama-3.1 | 8B | 8B | 98 | 114 | 63 | 65 |
| gpt-oss | 21B | 3.6B | 149 | 163 | 143 | 136 |
| Qwen3.6 | 27B | 27B | 25 | 33 | 18 | 19 |
| Devstral-Small-2 | 24B | 24B | 35 | 38 | 22 | 22 |
| Qwen3-Coder-Next | 80B | 3B | 87 | 106 | 71 | 83 |

**Quant**: Q4 is always faster than Q8 for the same model+runtime — 42–80%
faster for the three dense models. gpt-oss barely moves (+4–20%)
because its MoE experts are natively MXFP4 regardless of the nominal quant
label, so Q4_K_M and Q8_0 land within ~500MB of each other on disk.
Devstral-Small-2 and Qwen3-Coder-Next fall in between (+0–22%).

**Runtime**: MLX beats llama.cpp at the same quant in 10 of 12 combos, by
+2% to +31%. The exceptions are gpt-oss Q8 and Devstral-Small-2 Q8,
where llama.cpp is ~5% and ~3% faster respectively.
Qwen3-Coder-Next, the only other MoE model besides gpt-oss, backs up
the same pattern: at Q4 MLX it generates close to Llama-3.1's speed (106
vs. 114 tok/s) despite carrying ~10x the total parameters — active
parameter count, not total size, governs MoE generation speed.

### Prompt processing speed

Tokens/sec ingesting a fresh 512-token prompt (`pp512`) — how fast a model
chews through new context (your codebase, a long message) before it starts
responding:

| model | params | active | Q4 llama.cpp | Q4 MLX | Q8 llama.cpp | Q8 MLX |
|---|---|---|---|---|---|---|
| Qwen3 | 4B | 4B | 4782 | 4860 | 4475 | 4694 |
| Llama-3.1 | 8B | 8B | 2714 | 3019 | 2699 | 2474 |
| gpt-oss | 21B | 3.6B | 3243 | 4018 | 3298 | 3865 |
| Qwen3.6 | 27B | 27B | 669 | 571 | 701 | 653 |
| Devstral-Small-2 | 24B | 24B | 1014 | 889 | 1013 | 749 |
| Qwen3-Coder-Next | 80B | 3B | 2124 | 2066 | 2029 | 1875 |

**Quant**: barely matters here. Same-runtime Q4 vs Q8 differ by low
single-digit percents almost everywhere — the opposite of generation.
Devstral-Small-2 is the outlier: MLX drops ~16% from Q4 to Q8.

**Runtime**: a closer contest than for generation. MLX wins 5 of 12 combos;
llama.cpp takes the rest — Qwen3.6 at both quants (+7–17%),
Llama-3.1 at Q8 (+9%), and now Devstral-Small-2 (+14–35%) and
Qwen3-Coder-Next (+3–8%) at both quants. No runtime is simply "faster"
here — it depends on model and quant.

### Context depth tax

Prompt-processing slowdown at 32K tokens of prior context vs. an empty one:

| model | params | active | Q4 llama.cpp | Q4 MLX | Q8 llama.cpp | Q8 MLX |
|---|---|---|---|---|---|---|
| Qwen3 | 4B | 4B | 11.8x | 5.1x | 11.0x | 4.9x |
| Llama-3.1 | 8B | 8B | 6.2x | 3.2x | 6.3x | 2.8x |
| gpt-oss | 21B | 3.6B | 3.5x | 1.8x | 3.6x | 1.7x |
| Devstral-Small-2 | 24B | 24B | 3.6x | 1.9x | 3.8x | 1.8x |
| Qwen3.6 | 27B | 27B | 2.1x | 1.1x | 2.2x | 1.4x |
| Qwen3-Coder-Next | 80B | 3B | 2.4x | 1.8x | 2.3x | 1.7x |

Counterintuitive: the largest models (Qwen3.6, Qwen3-Coder-Next)
degrade *least* with depth, and the smallest (Qwen3) degrades *most* —
the opposite of what raw size would predict. The likely reason is
architecture, not size (see the README's Attention variant section):
Qwen3.6 and Qwen3-Coder-Next both replace three out of every four
attention layers with a linear-attention variant whose cost doesn't grow
with context at all, and gpt-oss's sliding-window layers cap how much
of the cache half its layers ever look at. Qwen3 and Llama-3.1, by
contrast, use plain grouped-query attention on every
layer, so their cost grows uniformly with depth with nowhere to hide it.
Devstral-Small-2 complicates a purely architectural story, though: it's
also plain grouped-query attention, yet its tax (3.6–3.8x) sits far below
Qwen3's and Llama-3.1's — closer to gpt-oss's. Architecture looks
like the dominant factor, but raw size still seems to matter some within a
given attention type.

Matters for long agentic coding sessions — response latency creeps up well
before the context window fills, and smaller models pay for it more.

### GPU vs CPU

Speedup from Metal offload (`-ngl 99`) vs CPU-only (`-ngl 0`), llama.cpp
only — MLX has no CPU-only mode to compare against (its whole premise is
unified-memory GPU execution):

| model | params | active | pp512, Q4 | pp512, Q8 | tg128, Q4 | tg128, Q8 |
|---|---|---|---|---|---|---|
| Qwen3 | 4B | 4B | 31.1x | 22.5x | 2.9x | 1.9x |
| Llama-3.1 | 8B | 8B | 30.3x | 24.1x | 3.3x | 2.1x |
| gpt-oss | 21B | 3.6B | 29.6x | 29.0x | 2.5x | 2.3x |
| Devstral-Small-2 | 24B | 24B | 35.7x | 30.7x | 3.5x | 2.3x |
| Qwen3.6 | 27B | 27B | 26.8x | 23.7x | 3.1x | 2.1x |
| Qwen3-Coder-Next | 80B | 3B | 20.6x | 17.0x | 2.2x | 2.0x |

GPU acceleration is not optional here: 17–36x faster prompt processing,
1.9–3.5x faster generation (generation is less parallel to begin with, so
gains less). Consistent across all six models. Qwen3-Coder-Next sees the
smallest GPU speedup of the roster on both metrics — plausibly because its
MoE routing (3B active of 80B total) already does less matrix-multiply work
per token even on CPU, leaving less on the table for Metal to parallelize
away.

**Thermal**: no meaningful throttling either. Sustained 1500-token
generations land within ~5% rep-to-rep for every model tested, including
both new models. See the README's Thermal/sustained-load caveat section
for the methodology.

### What this means in practice for coding

Coding work tends to fall into a few recurring patterns. Here's what the
numbers above translate to for each.

**A single quick edit or function** — "write me a function that does X" —
is one generation, no accumulated context. Nobody reviews code line-by-line
as it streams; you wait for the complete function or diff, then read or run
it. So tok/s translates directly into wall-clock wait before there's
something to review. A ~500-token reply (a function plus a short
explanation):

| config | tok/s | time |
|---|---|---|
| Qwen3, Q4 MLX (fastest) | 184 | 2.7s |
| gpt-oss, Q4 MLX | 163 | 3.1s |
| Llama-3.1, Q4 MLX | 114 | 4.4s |
| Qwen3.6, Q8 llama.cpp (slowest) | 18 | 27.9s |

Fast end: instant. Slow end: long enough to switch tabs and come back.

**An agentic loop** — read a file, edit it, run the tests, repeat —
accumulates context every turn instead of starting fresh each time. Both
halves of a turn slow down as that context grows: the model has to
re-process everything already in context before it can respond, and
generation itself degrades too (see [Context depth tax](#context-depth-tax)).
On Qwen3 (Q4, MLX), one turn — 512 tokens in,
500 out — costs ~2.8s early in a session and ~7.9s once you're 32K tokens
deep. Qwen3.6 only goes from ~15.9s to ~18.9s over the same span (it
degrades least with depth) but starts much slower to begin with either way.
A long-running agent gets measurably slower turn over turn, well before it
runs out of context window.

**Pointing a model at a large file or codebase for the first time** is pure
prompt processing — dead time, nothing streams back yet. Extrapolating from
`pp512` throughput, an 8K-token file takes ~1.7s to ingest on the fastest
config here (Qwen3, Q4 MLX) vs ~12.2s on the slowest (Qwen3.6, Q4
llama.cpp).

**A long unattended run** — the kind of multi-hour autonomous coding
session agentic tools increasingly get used for — doesn't get slower from
heat on this hardware: sustained 1500-token generations stayed within ~5%
of their starting throughput (see the README's Thermal/sustained-load
caveat section). Context growth, not thermal throttling, is what slows an
unattended agent down over a long session.

### Local vs. a frontier cloud model (e.g. Claude Code)

Everything above is one Mac running inference by itself. The natural
comparison is a hosted frontier model over the network — Claude, or
whatever you're using through Claude Code. This repo doesn't benchmark
that side — it's local-only, Apple Silicon macOS (see the README's
Requirements) — so there's no tok/s number to put next to the tables
above. The tradeoffs are still worth naming:

- **No hardware ceiling to hit.** A cloud provider runs on datacenter-scale
  hardware, not one Mac's unified memory and thermal envelope. The
  context-depth tax and the thermal check above are specifically local
  constraints — using a frontier model over the network, you don't
  personally hit them, even though the provider deals with the equivalent
  at their own scale.
- **Network adds its own latency.** Local inference has no round trip; a
  cloud request adds network time and provider-side queueing on top of
  generation time — variance a local model never has, but also not
  something this benchmark measures.
- **Frontier models are larger.** Every model tested here, including
  Qwen3-Coder-Next's 80B total (3B of it active per token), is well below
  frontier scale, so the quality half of this benchmark's tradeoff likely
  favors cloud — by how much isn't something this repo can answer without
  testing it directly.
- **Cost and privacy cut the other way.** Local inference has no per-token
  cost and keeps code on-device. Cloud usage means paying per token and
  sending code and context to a third party.

## Quality findings

Full data (generated locally, not committed): `eval_results.jsonl` (raw,
every run), `eval_runs/<run_id>/` (per-problem details).

### HumanEval vs. HumanEval+

Pass@1, averaged across quants/runtimes, fast mode:

| model | params | active | HumanEval | HumanEval+ |
|---|---|---|---|---|
| Qwen3 | 4B | 4B | 100% | 80% |
| Llama-3.1 | 8B | 8B | 80% | 55% |
| gpt-oss | 21B | 3.6B | 97.5% | 77.5% |
| Qwen3.6 | 27B | 27B | 100% | 78.75% |
| Devstral-Small-2 | 24B | 24B | 92.5% | 78.75% |
| Qwen3-Coder-Next | 80B | 3B | 95% | 86.25% |

Base HumanEval mostly saturates — 4 of 6 models score 92.5% or higher, two
hitting exactly 100% — leaving little room to see a quality difference
except at the bottom. HumanEval+'s stricter tests restore a real spread
(45–90% per run). Llama-3.1 is the outlier, 23–31 points behind every
other model (its best run just ties the pack's worst); the rest cluster
together despite a wide size range (4B–80B total). Qwen3-Coder-Next posts
the best HumanEval+ score of the roster, coding-specialized training
plausibly paying off on the harder variant specifically.

### Thinking mode

Qwen3.6, seconds per problem (same hard problem set both ways):

| quant / runtime | fast | thinking | multiplier |
|---|---|---|---|
| 4bit, MLX | 13.7s | 174.6s | 12.7x |
| 8bit, MLX | 19.5s | 216.8s | 11.1x |
| Q4_K_M, llama.cpp | 14.9s | 214.6s | 14.4x |
| Q8_0, llama.cpp | 26.1s | 193.1s | 7.4x |

A 7–14x latency cost for an inconsistent quality return: pass rate moves up
for two quants and down for the other two (n=8, small-sample caveat
applies). gpt-oss's thinking mode never beat its own fast-mode pass
rate in any quant/runtime combo tested — matched at best, worse the rest of
the time.

Qwen3-Coder-Next and Devstral-Small-2 tell a different story: enabling
`thinking` barely moves their latency (1.1–2.0x, vs. 7–14x for
Qwen3.6), the same "no-op on a non-thinking release" pattern the README's
Fast mode vs. thinking mode section notes for `Qwen3-4B-Instruct-2507` —
both are coding-specialized releases that likely were never trained with a
distinct reasoning mode. Pass rate
tracks that weak latency signal: Qwen3-Coder-Next's thinking-mode pass rate
is a flat 87.5% across all four quant/runtime combos regardless of its
fast-mode score (80–90%), while Devstral-Small-2 swings in both directions
and mostly down (−12.5, −5, +2.5, −12.5 points) — noisy rather than a clear
win either way, on the same small-sample caveat as above.
