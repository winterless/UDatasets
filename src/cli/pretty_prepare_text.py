#!/usr/bin/env python3
"""
Pretty print prepare jsonl records that use <|role|> blocks and <|tool_call|>.

Example:
  python src/cli/pretty_prepare_text.py -i out/prepare/Toucan-1.5M/000_00000.jsonl -n 1
  python src/cli/pretty_prepare_text.py -i out/prepare/agent-data-collection/000_00000.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from textwrap import indent


TAG_RE = re.compile(r"<\|([^|>]+)\|>")
TOOL_CALL_END = "</|tool_call|>"


def split_tagged_text(text: str) -> list[tuple[str, str]]:
    """
    Split text into (role, content) blocks.
    Supports <|role|>... blocks and <|tool_call|>...</|tool_call|>.
    """
    parts: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        m = TAG_RE.search(text, i)
        if not m:
            tail = text[i:]
            if tail.strip():
                parts.append(("text", tail))
            break
        tag = m.group(1)
        if m.start() > i:
            pre = text[i : m.start()]
            if pre.strip():
                parts.append(("text", pre))
        i = m.end()
        if tag == "tool_call":
            end = text.find(TOOL_CALL_END, i)
            if end == -1:
                content = text[i:]
                i = n
            else:
                content = text[i:end]
                i = end + len(TOOL_CALL_END)
            parts.append(("tool_call", content))
        else:
            m2 = TAG_RE.search(text, i)
            if m2:
                content = text[i : m2.start()]
                i = m2.start()
            else:
                content = text[i:]
                i = n
            parts.append((tag, content))
    return parts


def pretty_text(text: str) -> str:
    if "<|" not in text:
        return indent(text.strip(), "  ")
    blocks = split_tagged_text(text)
    out = []
    for role, content in blocks:
        content_s = content.strip()
        if not content_s:
            continue
        out.append(f"[{role}]")
        out.append(indent(content_s, "  "))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty print prepare jsonl")
    parser.add_argument("-i", "--input", required=True, help="Path to jsonl file")
    parser.add_argument("-n", "--num-records", type=int, default=None, help="Print first N records")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("uuid") or record.get("id") or f"line:{idx}"
            text = record.get("text", "") or ""
            print(f"=== Record {idx} | id={doc_id} ===")
            print(pretty_text(text))
            print()
            if args.num_records is not None and idx >= args.num_records:
                break


if __name__ == "__main__":
    main()
