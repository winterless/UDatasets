#!/usr/bin/env python3
"""
Postprocess augmentation for prepare outputs.

Currently supports:
  - tool_mcq: inject tool-selection and param MCQs before tool_call blocks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.postprocess import run_postprocess_tool_mcq


def main() -> None:
    ap = argparse.ArgumentParser(description="Postprocess prepare outputs")
    ap.add_argument("-i", "--input", required=True, help="Input prepare root directory")
    ap.add_argument("-o", "--output", required=True, help="Output root directory")
    ap.add_argument("--kind", default="tool_mcq", choices=["tool_mcq"], help="Postprocess kind")
    ap.add_argument("--dataset", default="", help="Only process a single dataset subdir")
    ap.add_argument("--max-mcq", type=int, default=3, help="Max MCQ blocks per record")
    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"{in_root} not found")

    datasets = [in_root / args.dataset] if args.dataset else [p for p in in_root.iterdir() if p.is_dir()]
    for ds in datasets:
        out_dir = out_root / ds.name
        if args.kind == "tool_mcq":
            run_postprocess_tool_mcq(ds, out_dir, max_mcq=int(args.max_mcq))


if __name__ == "__main__":
    main()
