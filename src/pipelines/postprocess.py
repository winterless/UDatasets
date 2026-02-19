from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable


TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "</|tool_call|>"


def _extract_tools_from_text(text: str) -> list[dict]:
    """
    Try to extract a list of tool specs from text.
    Supports blocks like:
      <tools> ... </tools> (each line is a JSON object)
      or a JSON list inside the block.
    """
    tools: list[dict] = []
    start = text.find("<tools>")
    end = text.find("</tools>")
    if start != -1 and end != -1 and end > start:
        raw = text[start + len("<tools>") : end].strip()
        # Try JSON list
        if raw.startswith("["):
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return [t for t in data if isinstance(t, dict)]
            except Exception:
                pass
        # Try JSON object per line
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


def _tool_required_params(tool: dict) -> list[str]:
    func = tool.get("function")
    if isinstance(func, dict):
        params = func.get("parameters")
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                return [str(x) for x in req if isinstance(x, (str, int))]
    return []


def _parse_tool_call_payload(payload: str) -> dict | None:
    payload = payload.strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # fallback: try single quotes -> double quotes
        try:
            obj = json.loads(payload.replace("'", "\""))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _render_mcq_tool(name: str, options: list[str], correct_idx: int) -> str:
    letters = "ABCD"
    lines = [
        "<|mcq_tool|>",
        "Instruction: Choose the next tool to call based on the dialogue context.",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i]}) {opt}")
    lines.append(f"Answer: {letters[correct_idx]}")
    lines.append("</|mcq_tool|>")
    return "\n".join(lines)


def _render_mcq_params(required: list[str], options: list[str], correct_idx: int) -> str:
    letters = "ABCD"
    req = ", ".join(required) if required else "(none)"
    lines = [
        "<|mcq_params|>",
        f"Instruction: Choose the required parameters for the tool call (Required: {req}).",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i]}) {opt}")
    lines.append(f"Answer: {letters[correct_idx]}")
    lines.append("</|mcq_params|>")
    return "\n".join(lines)


def augment_tool_mcq_record(text: str, *, seed: int = 0, max_mcq: int = 3) -> str:
    tools = _extract_tools_from_text(text)
    tool_names = [n for n in (_tool_name(t) for t in tools) if n]

    rng = random.Random(seed)
    out = []
    pos = 0
    mcq_count = 0
    # fallback: gather tool names from existing tool_call payloads in this record
    if not tool_names:
        tmp_names = set()
        scan_pos = 0
        while True:
            start = text.find(TOOL_CALL_START, scan_pos)
            if start == -1:
                break
            end = text.find(TOOL_CALL_END, start)
            if end == -1:
                break
            payload = text[start + len(TOOL_CALL_START) : end]
            call = _parse_tool_call_payload(payload)
            if call and isinstance(call.get("name"), str):
                tmp_names.add(call["name"])
            scan_pos = end + len(TOOL_CALL_END)
        tool_names = sorted(tmp_names)
    while True:
        start = text.find(TOOL_CALL_START, pos)
        if start == -1:
            out.append(text[pos:])
            break
        end = text.find(TOOL_CALL_END, start)
        if end == -1:
            out.append(text[pos:])
            break
        payload = text[start + len(TOOL_CALL_START) : end]
        call = _parse_tool_call_payload(payload)
        out.append(text[pos:start])
        if call and mcq_count < max_mcq:
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            if name and name in tool_names and len(tool_names) >= 2:
                distractors = [n for n in tool_names if n != name]
                rng.shuffle(distractors)
                options = [name] + distractors[:3]
                rng.shuffle(options)
                correct_idx = options.index(name)
                out.append(_render_mcq_tool(name, options, correct_idx))

                # params MCQ
                tool = next((t for t in tools if _tool_name(t) == name), None)
                if tool:
                    req = _tool_required_params(tool)
                    if req:
                        wrong1 = ", ".join(req[:-1]) if len(req) > 1 else "(none)"
                        wrong2 = ", ".join(req + ["extra"]) if req else "extra"
                        wrong3 = "(none)"
                        opts = [", ".join(req), wrong1, wrong2, wrong3]
                        rng.shuffle(opts)
                        out.append(_render_mcq_params(req, opts, opts.index(", ".join(req))))
                mcq_count += 1
        out.append(text[start : end + len(TOOL_CALL_END)])
        pos = end + len(TOOL_CALL_END)
    return "".join(out)


def run_postprocess_tool_mcq(input_dir: Path, output_dir: Path, *, max_mcq: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(input_dir.glob("*.jsonl")):
        dst = output_dir / src.name
        with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as w:
            for idx, line in enumerate(f, 1):
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
                seed = hash(obj.get("uuid") or obj.get("id") or idx) & 0xFFFFFFFF
                obj["text"] = augment_tool_mcq_record(text, seed=seed, max_mcq=max_mcq)
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")


POSTPROCESS_REGISTRY: dict[str, Callable[..., None]] = {
    "tool_mcq": run_postprocess_tool_mcq,
}


def write_pretty_txt_for_dir(input_dir: Path, *, suffix: str = "_pretty.txt") -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return
    try:
        from cli.pretty_prepare_text import (
            _extract_tools_from_text,
            _tool_desc,
            _tool_name,
            _tool_required_params,
            pretty_text,
        )
    except Exception:
        return
    for src in sorted(input_dir.glob("*.jsonl")):
        dst = src.with_name(f"{src.stem}{suffix}")
        try:
            with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as w:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
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
                                lines.append(f"    {desc}")
                            if req:
                                lines.append(f"    required: {', '.join(req)}")
                    lines.append("Messages:")
                    lines.append("  " + pretty_text(text).replace("\n", "\n  "))
                    w.write("\n".join(lines) + "\n\n")
        except Exception:
            continue


def run_postprocess(kind: str, input_dir: Path, output_dir: Path, *, config: dict | None = None) -> None:
    fn = POSTPROCESS_REGISTRY.get(kind)
    if not fn:
        raise ValueError(f"Unknown postprocess kind: {kind}")
    config = config or {}
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        return

    def _run(in_dir: Path, out_dir: Path) -> None:
        if kind == "tool_mcq":
            fn(in_dir, out_dir, max_mcq=int(config.get("max_mcq", 1)))
            return
        fn(in_dir, out_dir)

    # Allow in-place stacking (input_dir == output_dir) safely via a temp dir.
    try:
        same_dir = input_dir.resolve() == output_dir.resolve()
    except Exception:
        same_dir = str(input_dir) == str(output_dir)

    if same_dir:
        tmp = output_dir.parent / f".{output_dir.name}.__tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        _run(input_dir, tmp)
        shutil.rmtree(output_dir, ignore_errors=True)
        tmp.replace(output_dir)
        return

    _run(input_dir, output_dir)

