"""Small, dependency-free helpers shared by bench.py, coding_eval.py, and
download_models.py. Deliberately stdlib-only (no mlx_lm, no third-party
imports) so importing this doesn't reintroduce the constraint it exists to
avoid: bench.py has to keep running under bare system python3, without
needing the project venv.

Not shared with mlx_bench/bench.py — that directory is a deliberately
self-contained piece of the benchmark suite (see mlx_bench/README.md), so it
keeps its own copies of the same small functions rather than reaching across
into scripts/.
"""
from pathlib import Path

_ROOT_MARKERS = (".git", "CONTRIBUTING.md")


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for the repo root, rather than assuming
    a fixed directory depth (e.g. `parents[1]`) — so this keeps working if a
    driver script ever moves to a different depth under the repo root."""
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(f"could not find repo root (no {_ROOT_MARKERS} found) walking up from {start}")


def slugify(*parts: str) -> str:
    return "_".join(p.replace(" ", "-") for p in parts if p)


def select_models(models: list[dict], filters: list[str] | None) -> list[dict]:
    """Filter a model roster by --model substrings, OR'd together, matched
    case-insensitively against "family name quant". No filters means the
    whole roster, preserving the old default behavior."""
    if not filters:
        return models
    needles = [f.lower() for f in filters]
    return [
        m for m in models
        if any(n in f"{m['family']} {m['name']} {m['quant']}".lower() for n in needles)
    ]
