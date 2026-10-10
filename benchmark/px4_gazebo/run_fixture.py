"""Run the dependency-free handoff fixture and write JSONL."""

from __future__ import annotations

from .benchmark_runner import main, run_episode


def run(seed: int) -> dict:
    return run_episode(seed)


if __name__ == "__main__":
    main()
