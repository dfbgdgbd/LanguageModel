"""Train the retrieval index from scratch using the expanded corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smalllm.config import DEFAULT_RETRIEVAL_CORPUS_PATH, DEFAULT_RETRIEVAL_INDEX
from smalllm.retrieval import train_retrieval_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_RETRIEVAL_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RETRIEVAL_INDEX)
    args = parser.parse_args()
    count, size = train_retrieval_index(args.corpus, args.output)
    print(f"Indexed {count:,} prompts into {args.output}")
    print(f"Index size: {size / 1_048_576:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
