#!/usr/bin/env python3
"""Regenerates data/humaneval.jsonl and data/humanevalplus.jsonl from their
upstream sources. These files are committed (coding_eval.py needs them at
runtime and they're small), but weren't previously reproducible from
anything in this repo — this script is that missing provenance, not
something you need to run day to day.

- data/humaneval.jsonl: openai/openai_humaneval's test-split parquet
  (Hugging Face dataset), converted to plain JSONL for stdlib `json`
  loading with no dataset-library dependency at eval time.
- data/humanevalplus.jsonl: evalplus/humanevalplus's test.jsonl, copied
  verbatim — it already ships as JSONL upstream, nothing to convert.

Verified to reproduce the currently-committed files byte-for-byte as of
this script's writing; upstream could of course revise either dataset
later, in which case a diff here is real signal, not a bug in this script.

Requires pyarrow to read the HumanEval parquet file (`pip install
pyarrow`) — not a dependency of the main benchmark suite, only of this
one-off refresh script, so it's not part of setup.sh's install.

Usage:
    .venv/bin/pip install pyarrow
    .venv/bin/python scripts/fetch_data.py
"""
import json
import sys
from pathlib import Path

from _common import find_repo_root

REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
DATA_DIR = REPO_ROOT / "data"

HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_FILE = "openai_humaneval/test-00000-of-00001.parquet"
HUMANEVALPLUS_REPO = "evalplus/humanevalplus"
HUMANEVALPLUS_FILE = "test.jsonl"

EXPECTED_PROBLEM_COUNT = 164


def fetch_humaneval() -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow required to read the HumanEval parquet file: pip install pyarrow")
    from huggingface_hub import hf_hub_download

    print(f"downloading {HUMANEVAL_REPO}:{HUMANEVAL_FILE} ...")
    path = hf_hub_download(repo_id=HUMANEVAL_REPO, repo_type="dataset", filename=HUMANEVAL_FILE)
    rows = pq.read_table(path).to_pylist()
    if len(rows) != EXPECTED_PROBLEM_COUNT:
        sys.exit(f"expected {EXPECTED_PROBLEM_COUNT} problems from {HUMANEVAL_REPO}, got {len(rows)}")

    out_path = DATA_DIR / "humaneval.jsonl"
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} problems to {out_path}")


def fetch_humanevalplus() -> None:
    from huggingface_hub import hf_hub_download

    print(f"downloading {HUMANEVALPLUS_REPO}:{HUMANEVALPLUS_FILE} ...")
    path = Path(hf_hub_download(repo_id=HUMANEVALPLUS_REPO, repo_type="dataset", filename=HUMANEVALPLUS_FILE))
    rows = [json.loads(line) for line in path.open()]
    if len(rows) != EXPECTED_PROBLEM_COUNT:
        sys.exit(f"expected {EXPECTED_PROBLEM_COUNT} problems from {HUMANEVALPLUS_REPO}, got {len(rows)}")

    out_path = DATA_DIR / "humanevalplus.jsonl"
    out_path.write_bytes(path.read_bytes())
    print(f"wrote {len(rows)} problems to {out_path}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetch_humaneval()
    fetch_humanevalplus()


if __name__ == "__main__":
    main()
