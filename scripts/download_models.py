#!/usr/bin/env python3
"""Downloads every model in a roster via `hf download`, driven entirely by
the roster file itself — adding a model means adding a [[models]] block
with hf_repo/hf_files, not editing this script or setup.sh.

For runtime="llama.cpp" rosters, each model's `hf_files` (one or more
filenames within `hf_repo`) land in `models/gguf/<family>/`, preserving any
subfolder structure the repo uses (e.g. Qwen3-Coder-Next's split Q8_0/
files). For runtime="mlx" rosters, the whole `hf_repo` is downloaded
straight into the model's own `path`.

Usage:
    .venv/bin/python scripts/download_models.py --models configs/models.toml
    .venv/bin/python scripts/download_models.py --models mlx_bench/configs/models.toml
    .venv/bin/python scripts/download_models.py --models configs/models.toml --model qwen3-coder-next
    .venv/bin/python scripts/download_models.py --models configs/models.toml --dry-run
"""
import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from _common import find_repo_root, select_models

REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
HF_BINARY = REPO_ROOT / ".venv" / "bin" / "hf"


def download_command(model: dict, runtime: str) -> list[str]:
    if runtime == "mlx":
        local_dir = REPO_ROOT / model["path"]
        return [str(HF_BINARY), "download", model["hf_repo"], "--local-dir", str(local_dir)]

    local_dir = REPO_ROOT / "models" / "gguf" / model["family"]
    return [str(HF_BINARY), "download", model["hf_repo"], *model["hf_files"], "--local-dir", str(local_dir)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True, help="path to a model roster .toml file")
    parser.add_argument(
        "--model",
        action="append",
        help="substring filter on 'family name quant' (case-insensitive), e.g. --model qwen3-coder-next. "
        "Repeatable, OR'd together. Omit to download the whole roster.",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the hf commands without running them")
    args = parser.parse_args()

    with args.models.open("rb") as f:
        config = tomllib.load(f)
    runtime = config.get("runtime", "llama.cpp")

    if not args.dry_run and not HF_BINARY.exists():
        sys.exit(f"hf CLI not found at {HF_BINARY} (did you run setup.sh's venv step first?)")

    models = select_models(config["models"], args.model)
    if not models:
        sys.exit(f"--model {args.model} matched no models in {args.models}")

    # Same (family, name) pair can appear multiple times (once per quant);
    # skip repeats of the exact same hf_repo+local-dir+files combination.
    seen: set[tuple] = set()
    for model in models:
        cmd = download_command(model, runtime)
        key = tuple(cmd[2:])
        if key in seen:
            continue
        seen.add(key)

        print(f"running: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
