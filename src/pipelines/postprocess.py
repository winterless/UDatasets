from __future__ import annotations

import json
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
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


def _tool_param_schema(tool: dict) -> tuple[list[str], dict]:
    func = tool.get("function")
    if not isinstance(func, dict):
        return ([], {})
    params = func.get("parameters")
    if not isinstance(params, dict):
        return ([], {})
    required = params.get("required") or []
    req_list = [str(x) for x in required if isinstance(x, (str, int))]
    props = params.get("properties") or {}
    return (req_list, props if isinstance(props, dict) else {})


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


def _parse_tool_call_arguments(call: dict | None) -> dict | None:
    if not isinstance(call, dict):
        return None
    args = call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
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


def _render_mcq_param_values(options: list[str], correct_idx: int) -> str:
    letters = "ABCD"
    lines = [
        "<|mcq_param_values|>",
        "Instruction: Choose the correct argument object for the tool call.",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i]}) {opt}")
    lines.append(f"Answer: {letters[correct_idx]}")
    lines.append("</|mcq_param_values|>")
    return "\n".join(lines)


def _format_args_json(args: dict) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        safe = {}
        for k in args.keys():
            try:
                json.dumps(args[k], ensure_ascii=False)
                safe[k] = args[k]
            except Exception:
                safe[k] = str(args[k])
        return json.dumps(safe, ensure_ascii=False)


def _build_required_params_options(required: list[str], rng: random.Random) -> tuple[list[str], str] | None:
    if not required:
        return None
    wrong1 = ", ".join(required[:-1]) if len(required) > 1 else "(none)"
    wrong2 = ", ".join(required + ["extra"])
    wrong3 = "(none)"
    opts = [", ".join(required), wrong1, wrong2, wrong3]
    rng.shuffle(opts)
    return (opts, ", ".join(required))


def _build_param_value_options(
    *,
    name: str,
    args: dict,
    req: list[str],
    props: dict,
    param_pool: ParamPool,
    rng: random.Random,
    max_param_values_neg: int,
) -> tuple[list[str], str] | None:
    if not args or not param_pool or not param_pool.enabled or max_param_values_neg <= 0:
        return None
    correct = _format_args_json(args)
    variations: set[str] = set()
    attempts = 0
    max_attempts = max_param_values_neg * 8
    fields = list(args.keys())
    if not fields:
        return None
    # choose a subset of fields to vary across options
    vary_fields = [f for f in fields if rng.random() < 0.6]
    if not vary_fields:
        vary_fields = [rng.choice(fields)]
    while len(variations) < max_param_values_neg and attempts < max_attempts:
        attempts += 1
        mutated = dict(args)
        k = rng.randint(1, min(2, len(vary_fields)))
        changed = 0
        for field in rng.sample(vary_fields, k):
            schema = props.get(field) if isinstance(props, dict) else None
            p_type = (
                schema.get("type")
                if isinstance(schema, dict) and isinstance(schema.get("type"), str)
                else _infer_param_type(args[field])
            )
            repl = param_pool.sample(name, field, p_type, args[field])
            if repl is not None and repl != args.get(field):
                mutated[field] = repl
                changed += 1
        if changed == 0:
            continue
        option = _format_args_json(mutated)
        if option and option != correct:
            variations.add(option)
    if not variations:
        return None
    options = list(variations)
    if len(options) > max_param_values_neg:
        options = rng.sample(options, max_param_values_neg)
    options.append(correct)
    rng.shuffle(options)
    return (options, correct)


def _infer_param_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str):
        return "string"
    return "unknown"


def _classify_string(value: str) -> str:
    lower = value.lower()
    if lower.startswith(("http://", "https://")):
        return "string:url"
    if lower.startswith(("file://", "s3://")) or "/" in value or "\\" in value:
        return "string:path_or_uri"
    if len(value.split()) >= 6:
        return "string:long_text"
    if value.isdigit():
        return "string:numeric"
    if any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value):
        return "string:alnum"
    return "string:general"


def _cluster_label(param_type: str, value: Any) -> str:
    if param_type in {"integer", "number"}:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            magnitude = len(str(abs(value))) if value != 0 else 1
            sign_bucket = "number:negative" if value < 0 else ("number:zero" if value == 0 else "number:positive")
            if magnitude <= 1:
                mag_bucket = "single_digit"
            elif magnitude <= 2:
                mag_bucket = "two_digits"
            elif magnitude <= 4:
                mag_bucket = "four_digits"
            elif magnitude <= 8:
                mag_bucket = "eight_digits"
            else:
                mag_bucket = "huge"
            return f"{sign_bucket}:{mag_bucket}"
        if isinstance(value, float):
            magnitude = abs(value)
            if magnitude == 0:
                return "number:zero"
            if magnitude < 1:
                return "number:fraction"
            if magnitude < 10:
                return "number:small"
            if magnitude < 100:
                return "number:medium"
            if magnitude < 1000:
                return "number:large"
            return "number:huge"
        return "number:other"
    if param_type == "boolean":
        return "boolean"
    if param_type == "array":
        return "array"
    if param_type == "object":
        return "object"
    if param_type == "string":
        return _classify_string(value) if isinstance(value, str) else "string:non_str"
    return "unknown"


def _add_cluster_value(entry: dict, cluster_key: str, value: Any, max_values: int) -> None:
    clusters = entry.setdefault("clusters", {})
    bucket = clusters.setdefault(cluster_key, {"count": 0, "values": [], "_seen": set()})
    bucket["count"] += 1
    if len(bucket["values"]) >= max_values:
        return
    key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if key in bucket["_seen"]:
        return
    bucket["_seen"].add(key)
    bucket["values"].append(value)


def _scrub_clusters(entry: dict) -> None:
    for bucket in (entry.get("clusters") or {}).values():
        bucket.pop("_seen", None)


class ParamPoolBuilder:
    def __init__(self, max_values: int = 120):
        self.max_values = max_values
        self.functions: dict[str, dict] = {}
        self.params: dict[str, dict] = {}
        self.types: dict[str, dict] = {}

    def register(self, func_name: str, arguments: dict, schema_props: dict | None = None) -> None:
        schema_props = schema_props or {}
        for param_name, value in arguments.items():
            schema = schema_props.get(param_name) if isinstance(schema_props, dict) else None
            p_type = _infer_param_type(value)
            if isinstance(schema, dict) and isinstance(schema.get("type"), str):
                p_type = schema.get("type") or p_type
            cluster = _cluster_label(p_type, value)

            func_entry = self.functions.setdefault(func_name, {"params": {}})
            param_entry = func_entry["params"].setdefault(
                param_name, {"type": p_type, "required": False, "observed": 0, "clusters": {}}
            )
            param_entry["observed"] += 1
            if p_type and not param_entry.get("type"):
                param_entry["type"] = p_type
            _add_cluster_value(param_entry, cluster, value, self.max_values)

            p_entry = self.params.setdefault(param_name, {"type": p_type, "observed": 0, "clusters": {}})
            p_entry["observed"] += 1
            if p_type and not p_entry.get("type"):
                p_entry["type"] = p_type
            _add_cluster_value(p_entry, cluster, value, self.max_values)

            t_entry = self.types.setdefault(p_type, {"observed": 0, "clusters": {}})
            t_entry["observed"] += 1
            _add_cluster_value(t_entry, cluster, value, self.max_values)

    def as_dict(self) -> dict:
        for func in self.functions.values():
            for param in func["params"].values():
                _scrub_clusters(param)
        for param in self.params.values():
            _scrub_clusters(param)
        for entry in self.types.values():
            _scrub_clusters(entry)
        return {"functions": self.functions, "params": self.params, "types": self.types}


class ParamPool:
    def __init__(self, data: dict | None):
        data = data or {}
        self.functions = data.get("functions") or {}
        self.params = data.get("params") or {}
        self.types = data.get("types") or {}

    @property
    def enabled(self) -> bool:
        return bool(self.functions or self.params or self.types)

    def sample(self, func_name: str, param_name: str, param_type: str | None, original: Any) -> Any | None:
        func_entry = (self.functions.get(func_name) or {}).get("params", {}).get(param_name)
        param_entry = self.params.get(param_name)
        type_entry = self.types.get(param_type) if param_type else None
        search_order = [func_entry, param_entry, type_entry]
        for entry in search_order:
            candidate = self._pick_alternative(entry, original)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _canonical(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    def _pick_alternative(self, entry: dict | None, original: Any) -> Any | None:
        if not entry:
            return None
        original_key = self._canonical(original)
        clusters = list((entry.get("clusters") or {}).values())
        random.shuffle(clusters)
        for cluster in clusters:
            values = cluster.get("values") or []
            if not values:
                continue
            pool = list(values)
            random.shuffle(pool)
            for value in pool:
                if self._canonical(value) != original_key:
                    return value
        return None


def load_param_pool(path: str | None) -> ParamPool:
    if not path:
        return ParamPool(None)
    p = Path(path)
    if not p.exists():
        return ParamPool(None)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ParamPool(data if isinstance(data, dict) else None)
    except Exception:
        return ParamPool(None)


_MCQ_PARAM_POOL: ParamPool | None = None


def _init_tool_mcq_worker(param_pool_path: str | None) -> None:
    global _MCQ_PARAM_POOL
    _MCQ_PARAM_POOL = load_param_pool(param_pool_path)


def _tool_mcq_process_file_core(
    src_path: Path,
    dst_path: Path,
    *,
    param_pool: ParamPool,
    max_param_values_neg: int,
    mcq_every_n: int | None,
) -> None:
    with src_path.open("r", encoding="utf-8") as f, dst_path.open("w", encoding="utf-8") as w:
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
            if mcq_every_n and mcq_every_n > 0:
                do_mcq = ((idx - 1) % mcq_every_n) == 0
                if do_mcq:
                    obj["text"] = augment_tool_mcq_record_all(
                        text,
                        seed=seed,
                        param_pool=param_pool,
                        max_param_values_neg=max_param_values_neg,
                    )
                else:
                    obj["text"] = text
            else:
                obj["text"] = augment_tool_mcq_record(
                    text,
                    seed=seed,
                    param_pool=param_pool,
                    max_param_values_neg=max_param_values_neg,
                )
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _tool_mcq_process_file_worker(
    src_path: str,
    dst_path: str,
    *,
    max_param_values_neg: int,
    mcq_every_n: int | None,
) -> None:
    param_pool = _MCQ_PARAM_POOL or ParamPool(None)
    _tool_mcq_process_file_core(
        Path(src_path),
        Path(dst_path),
        param_pool=param_pool,
        max_param_values_neg=max_param_values_neg,
        mcq_every_n=mcq_every_n,
    )


def _resolve_postprocess_workers(
    config: dict | None, *, file_count: int, workers_override: int | None = None
) -> int:
    cpu = int(os.cpu_count() or 1)
    workers = None
    if workers_override is not None and int(workers_override) > 0:
        workers = int(workers_override)
    if isinstance(config, dict):
        for key in ("workers", "postprocess_workers"):
            if key in config:
                try:
                    if workers is None:
                        workers = int(config.get(key))
                except Exception:
                    workers = None
                break
    if workers is None:
        workers = min(32, cpu)
    workers = max(1, min(int(workers), max(1, int(file_count))))
    return workers


def build_param_pool_from_prepare(input_dir: Path, output_path: Path, *, max_values: int = 120) -> None:
    builder = ParamPoolBuilder(max_values=max_values)
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    for src in sorted(input_dir.glob("*.jsonl")):
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text", "") or ""
                tools = _extract_tools_from_text(text)
                schema_map: dict[str, dict] = {}
                for t in tools:
                    name = _tool_name(t)
                    if not name:
                        continue
                    _, props = _tool_param_schema(t)
                    schema_map[name] = props
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
                    if call:
                        name = call.get("name") if isinstance(call.get("name"), str) else ""
                        args = _parse_tool_call_arguments(call)
                        if name and isinstance(args, dict) and args:
                            builder.register(name, args, schema_map.get(name))
                    scan_pos = end + len(TOOL_CALL_END)
    if output_path.is_dir() or output_path.suffix.lower() != ".json":
        output_path = output_path / "param_pool.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(builder.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def augment_tool_mcq_record(
    text: str,
    *,
    seed: int = 0,
    param_pool: ParamPool | None = None,
    max_param_values_neg: int = 3,
) -> str:
    tools = _extract_tools_from_text(text)
    tool_names = [n for n in (_tool_name(t) for t in tools) if n]
    tool_schema_map: dict[str, tuple[list[str], dict]] = {}
    for t in tools:
        name = _tool_name(t)
        if not name:
            continue
        tool_schema_map[name] = _tool_param_schema(t)

    rng = random.Random(seed)
    out = []
    pos = 0
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
        if call:
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            inserted = False
            if name and name in tool_names and len(tool_names) >= 2:
                distractors = [n for n in tool_names if n != name]
                rng.shuffle(distractors)
                options = [name] + distractors[:3]
                rng.shuffle(options)
                correct_idx = options.index(name)
                out.append(_render_mcq_tool(name, options, correct_idx))
                inserted = True

            # params MCQ (schema-based)
            tool = next((t for t in tools if _tool_name(t) == name), None)
            if tool:
                req = _tool_required_params(tool)
                if req:
                    built = _build_required_params_options(req, rng)
                    if built:
                        opts, correct_req = built
                        out.append(_render_mcq_params(req, opts, opts.index(correct_req)))
                    inserted = True
            # param_values MCQ (requires param_pool)
            args = _parse_tool_call_arguments(call)
            if args:
                req, props = tool_schema_map.get(name, ([], {}))
                built = _build_param_value_options(
                    name=name,
                    args=args,
                    req=req,
                    props=props,
                    param_pool=param_pool,
                    rng=rng,
                    max_param_values_neg=max_param_values_neg,
                )
                if built:
                    options, correct = built
                    out.append(_render_mcq_param_values(options, options.index(correct)))
                    inserted = True
        out.append(text[start : end + len(TOOL_CALL_END)])
        pos = end + len(TOOL_CALL_END)
    return "".join(out)


def augment_tool_mcq_record_all(
    text: str,
    *,
    seed: int = 0,
    param_pool: ParamPool | None = None,
    max_param_values_neg: int = 3,
) -> str:
    tools = _extract_tools_from_text(text)
    tool_names = [n for n in (_tool_name(t) for t in tools) if n]
    tool_schema_map: dict[str, tuple[list[str], dict]] = {}
    for t in tools:
        name = _tool_name(t)
        if not name:
            continue
        tool_schema_map[name] = _tool_param_schema(t)

    rng = random.Random(seed)
    out = []
    pos = 0
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
        if call:
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            # tool MCQ (only if multiple tools available)
            if name and name in tool_names and len(tool_names) >= 2:
                distractors = [n for n in tool_names if n != name]
                rng.shuffle(distractors)
                options = [name] + distractors[:3]
                rng.shuffle(options)
                correct_idx = options.index(name)
                out.append(_render_mcq_tool(name, options, correct_idx))

            tool = next((t for t in tools if _tool_name(t) == name), None)
            req = _tool_required_params(tool) if tool else []
            args = _parse_tool_call_arguments(call)
            req, props = tool_schema_map.get(name, (req, {}))

            # param_values MCQ preferred; fallback to required-params MCQ
            built_values = (
                _build_param_value_options(
                    name=name,
                    args=args,
                    req=req,
                    props=props,
                    param_pool=param_pool,
                    rng=rng,
                    max_param_values_neg=max_param_values_neg,
                )
                if args
                else None
            )
            if built_values:
                options, correct = built_values
                out.append(_render_mcq_param_values(options, options.index(correct)))
            else:
                built_req = _build_required_params_options(req, rng)
                if built_req:
                    opts, correct_req = built_req
                    out.append(_render_mcq_params(req, opts, opts.index(correct_req)))
        out.append(text[start : end + len(TOOL_CALL_END)])
        pos = end + len(TOOL_CALL_END)
    return "".join(out)


def run_postprocess_tool_mcq(
    input_dir: Path,
    output_dir: Path,
    *,
    param_pool_path: str | None = None,
    max_param_values_neg: int = 3,
    workers: int | None = None,
    mcq_every_n: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        return
    workers = workers or 1
    if workers <= 1 or len(files) <= 1:
        param_pool = load_param_pool(param_pool_path)
        for src in files:
            dst = output_dir / src.name
            _tool_mcq_process_file_core(
                src,
                dst,
                param_pool=param_pool,
                max_param_values_neg=max_param_values_neg,
                mcq_every_n=mcq_every_n,
            )
        return

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_tool_mcq_worker,
        initargs=(param_pool_path,),
    ) as ex:
        futs = []
        for src in files:
            dst = output_dir / src.name
            futs.append(
                ex.submit(
                    _tool_mcq_process_file_worker,
                    src.as_posix(),
                    dst.as_posix(),
                    max_param_values_neg=max_param_values_neg,
                    mcq_every_n=mcq_every_n,
                )
            )
        for fut in as_completed(futs):
            fut.result()


POSTPROCESS_REGISTRY: dict[str, Callable[..., None]] = {
    "tool_mcq": run_postprocess_tool_mcq,
    "param_pool": build_param_pool_from_prepare,
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


def run_postprocess(
    kind: str,
    input_dir: Path,
    output_dir: Path,
    *,
    config: dict | None = None,
    workers_override: int | None = None,
) -> None:
    fn = POSTPROCESS_REGISTRY.get(kind)
    if not fn:
        raise ValueError(f"Unknown postprocess kind: {kind}")
    config = config or {}
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        return

    def _run(in_dir: Path, out_dir: Path) -> None:
        print(f"[post] run_postprocess: kind={kind} in_dir={in_dir} out_dir={out_dir}", flush=True)
        if kind == "tool_mcq":
            pool_path = str(config.get("param_pool_path") or "").strip()
            if not pool_path:
                # Auto-detect when chained with param_pool in same output dir.
                candidate = (out_dir / "param_pool.json")
                if candidate.exists():
                    pool_path = candidate.as_posix()
                else:
                    candidate = (in_dir / "param_pool.json")
                    if candidate.exists():
                        pool_path = candidate.as_posix()
            file_count = len(list(in_dir.glob("*.jsonl")))
            workers = _resolve_postprocess_workers(
                config, file_count=file_count, workers_override=workers_override
            )
            mcq_every_n = int(config.get("mcq_every_n", 0)) if isinstance(config, dict) else 0
            fn(
                in_dir,
                out_dir,
                param_pool_path=pool_path,
                max_param_values_neg=int(config.get("max_param_values_neg", 3)),
                workers=workers,
                mcq_every_n=mcq_every_n,
            )
            return
        if kind == "param_pool":
            max_values = int(config.get("max_values", 120))
            fn(in_dir, out_dir, max_values=max_values)
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

