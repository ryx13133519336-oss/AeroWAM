"""CLI for recomputing metrics from raw episode logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import aggregate, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(read_jsonl(args.log), args.method)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
