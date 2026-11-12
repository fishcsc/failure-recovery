#!/usr/bin/env python3
"""Save the first quarter (by JSONL line/record count) of a JSONL dataset."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract the first quarter of a JSONL file")
    parser.add_argument("input", type=Path, help="Source JSONL path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSONL path")
    args = parser.parse_args()

    source = args.input
    destination = args.output or source.with_name(f"{source.stem}_first25pct.jsonl")
    if source.resolve() == destination.resolve():
        parser.error("Input and output must be different paths")

    with source.open("r", encoding="utf-8") as infile:
        total = sum(bool(line.strip()) for line in infile)
    keep = total // 4

    written = 0
    with source.open("r", encoding="utf-8") as infile, destination.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            if written >= keep:
                break
            outfile.write(line if line.endswith("\n") else line + "\n")
            written += 1

    print(f"Original records: {total}")
    print(f"Saved records: {written} (first 25%, rounded down)")
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()
