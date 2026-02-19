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


def _extract_tools_from_text(text: str) -> list[dict]:
    tools: list[dict] = []
    start = text.find("<tools>")
    end = text.find("</tools>")
    if start == -1 or end == -1 or end <= start:
        return tools
    raw = text[start + len("<tools>") : end].strip()
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [t for t in data if isinstance(t, dict)]
        except Exception:
            pass
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                tools.append(obj)
        except Exception:
            continue
    return tools


def _tool_name(tool: dict) -> str:
    func = tool.get("function")
    if isinstance(func, dict):
        name = func.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _tool_desc(tool: dict) -> str:
    func = tool.get("function")
    if isinstance(func, dict):
        desc = func.get("description")
        if isinstance(desc, str):
            return desc.strip()
    desc = tool.get("description")
    return desc if isinstance(desc, str) else ""


def _tool_required_params(tool: dict) -> list[str]:
    func = tool.get("function")
    if isinstance(func, dict):
        params = func.get("parameters")
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                return [str(x) for x in req if isinstance(x, (str, int))]
    return []


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


def _pretty_json_like(content: str) -> str:
    try:
        obj = json.loads(content)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return content.strip()


def pretty_text(text: str) -> str:
    if "<|" not in text:
        return indent(text.strip(), "  ")
    blocks = split_tagged_text(text)
    out = []
    for role, content in blocks:
        content_s = content.strip()
        if not content_s:
            continue
        if role in ("tool_call", "tool_output"):
            out.append(f"[{role}]")
            out.append(indent(_pretty_json_like(content_s), "  "))
        else:
            out.append(f"[{role}]")
            out.append(indent(content_s, "  "))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty print prepare jsonl")
    parser.add_argument("-i", "--input", required=True, help="Path to jsonl file")
    parser.add_argument("-n", "--num-records", type=int, default=None, help="Print first N records")
    parser.add_argument("-o", "--output", default="", help="Optional output file path")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("uuid") or record.get("id") or f"line:{idx}"
            text = record.get("text", "") or ""
            header = f"=== Record {idx} | id={doc_id} ==="
            tools = _extract_tools_from_text(text)
            lines = [header]
            if tools:
                lines.append("Available tools:")
                for t in tools:
                    name = _tool_name(t) or "unknown"
                    desc = _tool_desc(t)
                    req = _tool_required_params(t)
                    lines.append(f"  - {name}")
                    if desc:
                        lines.append(indent(desc, "    "))
                    if req:
                        lines.append(f"    required: {', '.join(req)}")
            lines.append("Messages:")
            lines.append(indent(pretty_text(text), "  "))
            output_text = "\n".join(lines) + "\n"
            if out_fh:
                out_fh.write(output_text + "\n")
            else:
                print(output_text)
            if args.num_records is not None and idx >= args.num_records:
                break
    if out_fh:
        out_fh.close()


if __name__ == "__main__":
    main()
