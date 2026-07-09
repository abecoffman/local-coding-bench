#!/usr/bin/env python3
"""Coding capability eval: runs each model/quant through a HumanEval subset,
scores pass@1 (greedy decoding, single sample), and appends a summary row to
eval_results.jsonl plus per-problem detail to eval_runs/<run_id>/details.jsonl.

This is the quality half of the speed-vs-quality tradeoff story; see
results.jsonl (bench.py / mlx_bench.py) for the speed half.

Caveats:
  - HumanEval subset, not the full 164 problems (see `n_problems` in the
    config) — a deliberate time/signal tradeoff, not full-benchmark rigor.
  - pass@1 with greedy decoding (temperature 0), single sample per problem.
    No pass@k / multi-sample estimation.
  - Code extraction (`extract_code`) expects a fenced ```python block
    containing `def {entry_point}`. Models that respond in an unexpected
    format (e.g. harmony-style channeled output) will fail extraction rather
    than fail the actual coding task — check eval_runs/<run_id>/details.jsonl
    for `error: "extraction_failed"` before concluding a model can't code.
  - Executes model-generated code directly via subprocess with a timeout, no
    further sandboxing. Acceptable for this local, non-adversarial context
    only.

Usage:
    .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label m5-max-macbook
    .venv/bin/python scripts/coding_eval.py --models configs/models-mlx.toml --sweep configs/sweeps/coding-eval.toml --label m5-max-macbook
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_RESULTS_PATH = REPO_ROOT / "eval_results.jsonl"
EVAL_RUNS_DIR = REPO_ROOT / "eval_runs"
DATASETS = {
    "humaneval": REPO_ROOT / "data" / "humaneval.jsonl",
    "humanevalplus": REPO_ROOT / "data" / "humanevalplus.jsonl",
}
# HumanEval+ test harnesses import numpy. Always execute candidate code with
# this interpreter rather than sys.executable, so results don't depend on
# which Python launched coding_eval.py (bare system python3 for the
# llama.cpp path has no numpy and would otherwise fail every problem).
TEST_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LLAMA_SERVER_BINARY = REPO_ROOT / "runtimes" / "llama.cpp" / "build" / "bin" / "llama-server"

INSTRUCTION_TEMPLATE = """Complete the following Python function. Respond with ONLY the complete function implementation in a single ```python code block, including the function signature and any necessary imports. Do not include any explanation, tests, or extra text.

{prompt}"""

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def slugify(*parts: str) -> str:
    return "_".join(p.replace(" ", "-") for p in parts if p)


def model_ready(model_path: Path) -> bool:
    """True if model_path is a complete GGUF file or a fully-downloaded MLX
    model dir. A bare .exists() is not enough for MLX dirs: hf download
    creates the directory immediately, so a still-downloading model would
    otherwise look "ready" and crash mid-eval on missing safetensors."""
    if model_path.is_file():
        return model_path.exists()
    if not model_path.is_dir():
        return False
    if not (model_path / "config.json").exists():
        return False
    if not any(model_path.glob("*.safetensors")):
        return False
    if any(model_path.rglob("*.incomplete")):
        return False
    return True


def load_problems(n: int, dataset: str = "humaneval", offset: int = 0) -> list[dict]:
    path = DATASETS[dataset]
    problems = [json.loads(line) for line in path.open()]
    return problems[offset : offset + n]


def extract_code(text: str, entry_point: str) -> str | None:
    # Last block, not first: thinking-mode responses can contain several
    # draft code blocks before the final one.
    matches = CODE_BLOCK_RE.findall(text)
    code = matches[-1] if matches else text
    if f"def {entry_point}" not in code:
        return None
    return code


def run_test(code: str, test: str, entry_point: str, timeout: int = 10) -> tuple[bool, str]:
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        path = Path(f.name)
    try:
        proc = subprocess.run(
            [str(TEST_PYTHON), str(path)], capture_output=True, timeout=timeout, text=True
        )
        return proc.returncode == 0, ("" if proc.returncode == 0 else proc.stderr[-500:])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        path.unlink(missing_ok=True)


def eval_problem(problem: dict, generate_fn, max_tokens: int) -> dict:
    instruction = INSTRUCTION_TEMPLATE.format(prompt=problem["prompt"])
    t0 = time.time()
    try:
        text = generate_fn(instruction, max_tokens)
    except Exception as e:
        return {"task_id": problem["task_id"], "passed": False, "error": f"generation_error: {e}"}
    gen_time_s = time.time() - t0

    code = extract_code(text, problem["entry_point"])
    if code is None:
        return {
            "task_id": problem["task_id"],
            "passed": False,
            "error": "extraction_failed",
            "gen_time_s": gen_time_s,
            "raw_response": text[:2000],
        }

    passed, err = run_test(code, problem["test"], problem["entry_point"])
    return {
        "task_id": problem["task_id"],
        "passed": passed,
        "error": err,
        "gen_time_s": gen_time_s,
    }


# --- llama.cpp backend: start llama-server once per model, use HTTP chat completions ---


def wait_for_server(port: int, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s")


def llama_server_generate(port: int, prompt: str, max_tokens: int, thinking: bool) -> str:
    # Thinking/reasoning models otherwise spend the whole token budget on
    # hidden reasoning and never emit `content`. enable_thinking covers
    # Qwen3.x; reasoning_effort covers gpt-oss's harmony format, which
    # ignores enable_thinking and needs its own knob turned down. No-op for
    # other templates either way.
    template_kwargs = (
        {"enable_thinking": True, "reasoning_effort": "high"}
        if thinking
        else {"enable_thinking": False, "reasoning_effort": "low"}
    )
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": template_kwargs,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    message = resp["choices"][0]["message"]
    # reasoning_content is where thinking-mode answers often live when the
    # model never transitions out of its reasoning phase within max_tokens.
    return message.get("reasoning_content", "") + "\n" + message.get("content", "")


def eval_llama_cpp_model(
    model_path: Path, problems: list[dict], max_tokens: int, port: int, thinking: bool
) -> list[dict]:
    proc = subprocess.Popen(
        [str(LLAMA_SERVER_BINARY), "-m", str(model_path), "--port", str(port), "--host", "127.0.0.1", "-ngl", "99"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(port)
        generate_fn = lambda prompt, mt: llama_server_generate(port, prompt, mt, thinking)  # noqa: E731
        return [eval_problem(p, generate_fn, max_tokens) for p in problems]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# --- MLX backend: load once, generate in-process ---


def eval_mlx_model(model_path: Path, problems: list[dict], max_tokens: int, thinking: bool) -> list[dict]:
    from mlx_lm.generate import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)
    thinking_kwargs = (
        {"enable_thinking": True, "reasoning_effort": "high"}
        if thinking
        else {"enable_thinking": False, "reasoning_effort": "low"}
    )

    def generate_fn(prompt, mt):
        messages = [{"role": "user", "content": prompt}]
        # See llama_server_generate's comment: covers both Qwen3.x
        # (enable_thinking) and gpt-oss harmony (reasoning_effort). Unlike
        # llama-server, mlx_lm returns the full raw text with no separate
        # reasoning_content field, so extract_code sees everything already.
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **thinking_kwargs
        )
        return mlx_generate(model, tokenizer, rendered, max_tokens=mt, sampler=sampler)

    return [eval_problem(p, generate_fn, max_tokens) for p in problems]


def run_model(model_cfg: dict, runtime: str, sweep_args: dict, sweep_name: str, label: str, port: int) -> None:
    model_path = REPO_ROOT / model_cfg["path"]
    if not model_ready(model_path):
        print(f"skip: model not ready (missing or still downloading): {model_path}", file=sys.stderr)
        return

    n_problems = sweep_args.get("n_problems", 20)
    max_tokens = sweep_args.get("max_tokens", 768)
    dataset = sweep_args.get("dataset", "humaneval")
    offset = sweep_args.get("problem_offset", 0)
    thinking = sweep_args.get("thinking", False)
    problems = load_problems(n_problems, dataset, offset)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{slugify(model_cfg['family'], model_cfg['name'], model_cfg['quant'])}"

    mode = "thinking" if thinking else "fast"
    print(
        f"evaluating: {model_cfg['name']} ({model_cfg['quant']}, {runtime}, {mode}) "
        f"on {dataset}[{offset}:{offset + n_problems}]"
    )
    t0 = time.time()
    if runtime == "llama.cpp":
        results = eval_llama_cpp_model(model_path, problems, max_tokens, port, thinking)
    elif runtime == "mlx":
        results = eval_mlx_model(model_path, problems, max_tokens, thinking)
    else:
        sys.exit(f"unsupported runtime: {runtime!r}")
    wall_s = time.time() - t0

    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / len(results) if results else 0.0

    run_dir = EVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "details.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "runtime": runtime,
        "run_id": run_id,
        "config": sweep_name,
        "label": label,
        "model_family": model_cfg["family"],
        "model_name": model_cfg["name"],
        "quant": model_cfg["quant"],
        "dataset": dataset,
        "problem_offset": offset,
        "thinking": thinking,
        "n_problems": len(results),
        "passed": passed,
        "pass_rate": pass_rate,
        "wall_s": wall_s,
        "test_time": datetime.now(timezone.utc).isoformat(),
    }
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_RESULTS_PATH.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"  -> pass_rate={pass_rate:.2%} ({passed}/{len(results)}), {wall_s:.1f}s, details in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="path to a model roster .toml file")
    parser.add_argument("--sweep", type=Path, required=True, help="path to a sweep-args .toml file")
    parser.add_argument("--label", default="", help="machine/environment tag (e.g. m5-max-macbook)")
    parser.add_argument("--port", type=int, default=8899, help="port for llama-server (llama.cpp runtime only)")
    args = parser.parse_args()

    with args.models.open("rb") as f:
        models_config = tomllib.load(f)
    with args.sweep.open("rb") as f:
        sweep_config = tomllib.load(f)

    runtime = models_config.get("runtime")
    if runtime not in ("llama.cpp", "mlx"):
        sys.exit(f"{args.models} must declare runtime = \"llama.cpp\" or \"mlx\", got {runtime!r}")

    sweep_name = args.sweep.stem
    sweep_args = sweep_config.get("args", {})

    for model_cfg in models_config["models"]:
        run_model(model_cfg, runtime, sweep_args, sweep_name, args.label, args.port)


if __name__ == "__main__":
    main()
