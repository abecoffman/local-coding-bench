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

LLAMA_CPP_COMMIT="a646006f0"
TRANSFORMERS_VERSION="5.0.0"

MODELS="all"
for arg in "$@"; do
  case "$arg" in
    --models=*) MODELS="${arg#*=}" ;;
    --models) shift ;;
    --skip-models) MODELS="none" ;;
  esac
done

want_model() {
  [ "$MODELS" = "all" ] && return 0
  [ "$MODELS" = "none" ] && return 1
  [[ ",$MODELS," == *",$1,"* ]]
}

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
.venv/bin/pip install -q mlx-lm "transformers==$TRANSFORMERS_VERSION" numpy
# mlx-lm declares transformers>=5.0.0 but is broken against anything past
# 5.0.0 as of mlx-lm 0.31.3 (an upstream bug, not a typo) — the exact pin
# above is required, not just a floor.

if [ "$MODELS" = "none" ]; then
  echo "=== skipping model downloads (--skip-models) ==="
else
  echo "=== models ($MODELS) ==="
  want_model qwen3-4b && {
    .venv/bin/hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf Qwen3-4B-Instruct-2507-Q8_0.gguf --local-dir models/gguf/qwen
    .venv/bin/hf download mlx-community/Qwen3-4B-Instruct-2507-4bit --local-dir models/mlx/qwen/Qwen3-4B-Instruct-2507-4bit
    .venv/bin/hf download mlx-community/Qwen3-4B-Instruct-2507-8bit --local-dir models/mlx/qwen/Qwen3-4B-Instruct-2507-8bit
  }
  want_model llama-3.1-8b && {
    .venv/bin/hf download unsloth/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --local-dir models/gguf/llama
    .venv/bin/hf download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --local-dir models/mlx/llama/Meta-Llama-3.1-8B-Instruct-4bit
    .venv/bin/hf download mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --local-dir models/mlx/llama/Meta-Llama-3.1-8B-Instruct-8bit
  }
  want_model gpt-oss-20b && {
    .venv/bin/hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf gpt-oss-20b-Q8_0.gguf --local-dir models/gguf/openai
    .venv/bin/hf download mlx-community/gpt-oss-20b-MXFP4-Q4 --local-dir models/mlx/openai/gpt-oss-20b-MXFP4-Q4
    .venv/bin/hf download mlx-community/gpt-oss-20b-MXFP4-Q8 --local-dir models/mlx/openai/gpt-oss-20b-MXFP4-Q8
  }
  want_model qwen3.6-27b && {
    .venv/bin/hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf Qwen3.6-27B-Q8_0.gguf --local-dir models/gguf/qwen
    .venv/bin/hf download mlx-community/Qwen3.6-27B-4bit --local-dir models/mlx/qwen/Qwen3.6-27B-4bit
    .venv/bin/hf download mlx-community/Qwen3.6-27B-8bit --local-dir models/mlx/qwen/Qwen3.6-27B-8bit
  }
fi

echo "=== done ==="
echo "Try: python3 scripts/bench.py --models configs/models.toml --sweep configs/sweeps/standard.toml --label \$(hostname -s)"
