"""Pretty print the last N audit log entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "audit.log"


def tail_lines(path: Path, tail: int) -> Iterable[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return lines[-tail:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show recent audit log entries")
    parser.add_argument("--tail", type=int, default=10, help="Number of entries to display")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Path to audit log")
    args = parser.parse_args()

    for line in tail_lines(args.log, args.tail):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(line.strip())
            continue
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
