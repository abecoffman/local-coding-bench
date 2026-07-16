#!/bin/bash
# Bootstraps everything this repo needs that isn't checked into git:
# llama.cpp (built from source, pinned commit), a Python venv (mlx-lm +
# deps), and a model roster downloaded from Hugging Face.
#
# Usage:
#   ./setup.sh                              # everything, full model roster
#   ./setup.sh --models qwen3-4b,llama-3.1-8b   # only these model families
#   ./setup.sh --skip-models                # llama.cpp + venv only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$(uname)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This project requires Apple Silicon macOS: llama.cpp's speed here relies on Metal," >&2
  echo "and MLX (mlx_bench/, half the coding eval) only runs on Apple Silicon at all." >&2
  exit 1
fi

if ! command -v cmake >/dev/null; then
  echo "cmake not found — install it first (e.g. 'brew install cmake') to build llama.cpp." >&2
  exit 1
fi

if ! command -v python3 >/dev/null || ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "python3 >= 3.11 required (every driver script uses the stdlib tomllib module," >&2
  echo "which doesn't exist before 3.11) — found: $(command -v python3 >/dev/null && python3 --version || echo 'none')" >&2
  exit 1
fi

LLAMA_CPP_COMMIT="a646006f0"
TRANSFORMERS_VERSION="5.0.0"

MODELS="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --models=*) MODELS="${1#*=}" ;;
    --models) MODELS="$2"; shift ;;
    --skip-models) MODELS="none" ;;
  esac
  shift
done

echo "=== llama.cpp (commit $LLAMA_CPP_COMMIT) ==="
if [ ! -d runtimes/llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git runtimes/llama.cpp
fi
git -C runtimes/llama.cpp fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
git -C runtimes/llama.cpp checkout "$LLAMA_CPP_COMMIT"
cmake -B runtimes/llama.cpp/build -S runtimes/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build runtimes/llama.cpp/build --config Release -j

echo "=== Python venv ==="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -q mlx-lm "transformers==$TRANSFORMERS_VERSION" huggingface_hub numpy
# mlx-lm declares transformers>=5.0.0 but is broken against anything past
# 5.0.0 as of mlx-lm 0.31.3 (an upstream bug, not a typo) — the exact pin
# above is required, not just a floor.
#
# huggingface_hub isn't a direct mlx-lm/transformers dependency we'd
# otherwise get for free at a predictable version — it's what provides the
# `hf` CLI every download below relies on. Listed explicitly so it's not
# just an implicit side effect of whatever transformers happens to pull in.

if ! .venv/bin/hf --version >/dev/null 2>&1; then
  echo "hf CLI not found at .venv/bin/hf after installing huggingface_hub — can't download models." >&2
  echo "Try '.venv/bin/pip install --upgrade huggingface_hub' and re-run, or file an issue if that doesn't help." >&2
  exit 1
fi

if [ "$MODELS" = "none" ]; then
  echo "=== skipping model downloads (--skip-models) ==="
else
  echo "=== models ($MODELS) ==="
  # Fully config-driven: every model's hf_repo/hf_files live in
  # configs/models.toml and mlx_bench/configs/models.toml, so adding a
  # model never means touching this script — see download_models.py.
  MODEL_ARGS=()
  if [ "$MODELS" != "all" ]; then
    IFS=',' read -ra MODEL_TOKENS <<< "$MODELS"
    for token in "${MODEL_TOKENS[@]}"; do
      MODEL_ARGS+=(--model "$token")
    done
  fi
  # ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": macOS's default /bin/bash
  # is 3.2 (GPL licensing), where a bare "${arr[@]}" on an empty array under
  # `set -u` raises "unbound variable" instead of expanding to nothing.
  .venv/bin/python scripts/download_models.py --models configs/models.toml "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"
  .venv/bin/python scripts/download_models.py --models mlx_bench/configs/models.toml "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"
fi

echo "=== done ==="
echo "Try:"
echo "  python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label \$(hostname -s)"
echo "  .venv/bin/python mlx_bench/bench.py --models mlx_bench/configs/models.toml --sweep mlx_bench/configs/sweeps/standard.toml --label \$(hostname -s)"
echo "  .venv/bin/python scripts/coding_eval.py --models configs/models.toml --sweep configs/sweeps/coding-eval.toml --label \$(hostname -s)"
echo "See README.md's Quickstart section for more."
