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
    conv = raw.get("conversations")
    if isinstance(conv, list):
        lines = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content}".strip())
        text = "\n".join(lines)

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


BASE_ADAPTERS: dict[str, Callable] = {
    "toucan": toucan_adapter,
    "agent-data-collection": agent_data_collection_adapter,
    "glaive": glaive_adapter,
    "hephaestus": hephaestus_adapter,
}


