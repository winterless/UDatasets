from __future__ import annotations

"""
Dataset-specific adapter functions (NO datatrove imports).

These functions convert a raw record dict into a datatrove-compatible "Document dict":
  {"text": str, "id": str, "metadata": dict}

The adapters keep raw payload in metadata for traceability.
"""

import json
from typing import Any, Callable, Dict

JsonDict = Dict[str, Any]


def _ensure_text(text: str | None) -> str:
    t = (text or "").strip()
    return t if t else "[NO_TEXT]"


def _stable_id_from_path(path: str, id_in_file: int | str) -> str:
    return f"{path}/{id_in_file}"


def toucan_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    raw = dict(data)
    uuid = raw.get("uuid")
    doc_id = uuid if isinstance(uuid, str) and uuid else _stable_id_from_path(path, id_in_file)

    msg_field = raw.get("messages")
    text = ""
    if isinstance(msg_field, str):
        try:
            msgs = json.loads(msg_field)
        except Exception:
            msgs = None
        if isinstance(msgs, list):
            lines = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    lines.append(f"{role}: {content}".strip())
                fc = m.get("function_call")
                if fc is not None:
                    lines.append(f"{role}.function_call: {fc}")
            text = "\n".join(lines)
    elif isinstance(msg_field, list):
        lines = []
        for m in msg_field:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content}".strip())
        text = "\n".join(lines)

    if not text and isinstance(raw.get("question"), str):
        text = raw["question"]

    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "Toucan-1.5M", "raw": raw}}


def agent_data_collection_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    raw = dict(data)
    doc_id = raw.get("id")
    doc_id = doc_id if isinstance(doc_id, str) and doc_id else _stable_id_from_path(path, id_in_file)

    text = ""
    # Multi-route fallback because agent-data-collection mixes several schemas:
    # 1) {"conversations": [{"role": "...", "content": "..."}]}
    # 2) {"conversations": [{"from": "...", "value": "..."}]}
    # 3) {"content": [{"class_": "...", "source": "...", "content": "...", "description": "..."}]}  (db-style)
    def _to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    conv = raw.get("conversations")
    if isinstance(conv, list):
        lines: list[str] = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = _to_str(m.get("role") or m.get("from") or m.get("speaker") or m.get("name") or "").strip()
            content = _to_str(m.get("content") if "content" in m else (m.get("value") if "value" in m else m.get("text"))).strip()
            if content:
                lines.append(f"{role}: {content}".strip() if role else content)
        text = "\n".join(lines).strip()

    # Nemotron-style fallback: some datasets use top-level `messages` with {role, content}
    if not text:
        msgs = raw.get("messages")
        if isinstance(msgs, list):
            lines = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = _to_str(m.get("role") or m.get("from") or m.get("speaker") or m.get("name") or "").strip()
                content = _to_str(m.get("content") if "content" in m else (m.get("value") if "value" in m else m.get("text"))).strip()
                if content:
                    lines.append(f"{role}: {content}".strip() if role else content)
            text = "\n".join(lines).strip()

    # db-style fallback: top-level `content` is a list of step dicts containing `content`/`description`
    if not text:
        steps = raw.get("content")
        if isinstance(steps, list):
            lines = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                role = _to_str(s.get("source") or s.get("class_") or "").strip()
                main = _to_str(s.get("content")).strip()
                desc = _to_str(s.get("description")).strip()
                if main:
                    lines.append(f"{role}: {main}".strip() if role else main)
                if desc:
                    lines.append(f"{role}.description: {desc}".strip() if role else desc)
            text = "\n".join(lines).strip()

    # Last-ditch fallback: common single-field prompts/questions.
    if not text:
        for k in ("question", "prompt", "instruction", "text", "input"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break

    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "agent-data-collection", "raw": raw}}


def glaive_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    raw = dict(data)
    doc_id = _stable_id_from_path(path, id_in_file)
    system = raw.get("system") if isinstance(raw.get("system"), str) else ""
    chat = raw.get("chat") if isinstance(raw.get("chat"), str) else ""
    text = (system + "\n\n" + chat).strip()
    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "glaive-function-calling-v2", "raw": raw}}


def hephaestus_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    raw = dict(data)
    rid = raw.get("id")
    doc_id = rid if isinstance(rid, str) and rid else _stable_id_from_path(path, id_in_file)
    text = raw.get("text") if isinstance(raw.get("text"), str) else ""
    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "Hephaestus-Forge", "raw": raw}}


def nemotron_math_v2_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    """
    Nemotron-Math-v2 schema (only fields we care about):
      {
        "uuid": "xx",
        "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "...", "reasoning_content": "..."}
        ],
        ...
      }

    Desired output:
      id = uuid
      text = user + content + assistant + reasoning_content + content
    We interpret that as a newline-joined sequence:
      "user\\n<user.content>\\nassistant\\n<assistant.reasoning_content>\\n<assistant.content>"
    (and we keep message order if multiple turns exist).
    """
    raw = dict(data)
    uuid = raw.get("uuid")
    doc_id = uuid if isinstance(uuid, str) and uuid else _stable_id_from_path(path, id_in_file)

    msgs = raw.get("messages")
    parts: list[str] = []
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            role_s = role.strip() if isinstance(role, str) else ""
            if role_s:
                parts.append(role_s)
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            if role_s == "assistant":
                rc = m.get("reasoning_content")
                if isinstance(rc, str) and rc.strip():
                    parts.append(rc.strip())
    text = "\n".join(parts).strip()
    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "Nemotron-Math-v2", "raw": raw}}


def nemotron_pretraining_sft_v1_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    """
    Nemotron-Pretraining-SFT-v1 schema (only fields we care about):
      {"id": "xx", "text": "xxxx", ...}

    Desired output:
      id = raw.id
      text = raw.text
    """
    raw = dict(data)
    rid = raw.get("id")
    doc_id = rid if isinstance(rid, str) and rid else _stable_id_from_path(path, id_in_file)
    text = raw.get("text") if isinstance(raw.get("text"), str) else ""
    return {"text": _ensure_text(text), "id": doc_id, "metadata": {"dataset": "Nemotron-Pretraining-SFT-v1", "raw": raw}}


BASE_ADAPTERS: dict[str, Callable] = {
    "toucan": toucan_adapter,
    "agent-data-collection": agent_data_collection_adapter,
    "glaive": glaive_adapter,
    "hephaestus": hephaestus_adapter,
    "nemotron-math-v2": nemotron_math_v2_adapter,
    "nemotron-pretraining-sft-v1": nemotron_pretraining_sft_v1_adapter,
}


