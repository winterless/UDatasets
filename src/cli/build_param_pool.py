#!/usr/bin/env python3
"""
Build a parameter-value pool from prepare JSONL outputs.

Example:
  PYTHONPATH=src python -m cli.build_param_pool \
    -i out_toucan_1/prepare/Toucan-1.5M \
    -o out_toucan_1/prepare/Toucan-1.5M/param_pool.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.postprocess import build_param_pool_from_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Build param pool from prepare JSONL outputs")
    parser.add_argument("-i", "--input-dir", required=True, help="Prepare dir containing *.jsonl")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path (or directory)")
    parser.add_argument("--max-values", type=int, default=120, help="Max values per cluster (default: 120)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    build_param_pool_from_prepare(input_dir, output_path, max_values=int(args.max_values))


if __name__ == "__main__":
    main()
