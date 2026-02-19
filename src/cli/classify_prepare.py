#!/usr/bin/env python3
"""
Classify prepare JSONL records into cpt/sft vs other based on role tags.

Criteria (default):
  - has both <|user|> and <|assistant|> -> cpt_sft
  - otherwise -> other

Example:
  python src/cli/classify_prepare.py -i out/prepare -o out/prepare_classified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify_text(text: str) -> str:
    has_user = "<|user|>" in text
    has_assistant = "<|assistant|>" in text
    return "cpt_sft" if (has_user and has_assistant) else "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify prepare jsonl records")
    ap.add_argument("-i", "--input", required=True, help="Prepare root directory")
    ap.add_argument("-o", "--output", required=True, help="Output root directory")
    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    if not in_root.exists():
        raise FileNotFoundError(f"{in_root} not found")

    total = {"cpt_sft": 0, "other": 0}
    for dataset_dir in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        for src in sorted(dataset_dir.glob("*.jsonl")):
            rel = src.relative_to(in_root)
            out_cpt = out_root / "cpt_sft" / rel
            out_other = out_root / "other" / rel
            out_cpt.parent.mkdir(parents=True, exist_ok=True)
            out_other.parent.mkdir(parents=True, exist_ok=True)

            with src.open("r", encoding="utf-8") as f, \
                out_cpt.open("w", encoding="utf-8") as w_cpt, \
                out_other.open("w", encoding="utf-8") as w_other:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    text = obj.get("text", "") or ""
                    bucket = classify_text(text)
                    if bucket == "cpt_sft":
                        w_cpt.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    else:
                        w_other.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total[bucket] += 1

    summary = out_root / "_summary.txt"
    summary.write_text(
        f"cpt_sft: {total['cpt_sft']}\nother: {total['other']}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
