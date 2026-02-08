from __future__ import annotations

"""
Dataset-specific adapter functions (NO datatrove imports).

These functions convert a raw record dict into a datatrove-compatible "Document dict":
  {"text": str, "id": str, "metadata": dict}

The adapters keep raw payload in metadata for traceability.
"""

import json
import re
from typing import Any, Callable, Dict

JsonDict = Dict[str, Any]


def _ensure_text(text: str | None) -> str:
    t = (text or "").strip()
    return t if t else ""


def _stable_id_from_path(path: str, id_in_file: int | str) -> str:
    return f"{path}/{id_in_file}"


def toucan_adapter(self, data: JsonDict, path: str, id_in_file: int | str) -> JsonDict:
    raw = dict(data)
    uuid = raw.get("uuid")
    doc_id = uuid if isinstance(uuid, str) and uuid else _stable_id_from_path(path, id_in_file)

    msg_field = raw.get("messages")
    text = ""
    def _has_duplicate_function_call(msgs: list[dict]) -> bool:
        seen: set[str] = set()
        for m in msgs:
            if not isinstance(m, dict):
                continue
            fc = m.get("function_call")
            if fc is None:
                continue
            if isinstance(fc, (dict, list)):
                try:
                    key = json.dumps(fc, sort_keys=True, ensure_ascii=False)
                except Exception:
                    key = str(fc)
            else:
                key = str(fc)
            if key in seen:
                return True
            seen.add(key)
        return False

    def _format_role_block(role: str, content: str) -> str:
        role_s = role.strip() if isinstance(role, str) else ""
        return f"<|{role_s}|>\n{content}".strip() if role_s else content

    def _format_tool_call(fc: object) -> str:
        if isinstance(fc, dict):
            try:
                payload = json.dumps(fc, ensure_ascii=False, sort_keys=True)
            except Exception:
                payload = str(fc)
        else:
            payload = str(fc)
        return f"<|tool_call|>\n{payload}\n</|tool_call|>"

    def _format_messages(msgs: list[dict]) -> str:
        parts: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "") or ""
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(_format_role_block(role, content.strip()))
            fc = m.get("function_call")
            if fc is not None:
                parts.append(_format_tool_call(fc))
            # Tool return content: prefer role "tool"/"function" if present.
            if role in ("tool", "function") and isinstance(content, str) and content.strip():
                parts[-1] = _format_role_block("tool_output", content.strip())
        return "\n\n".join(parts).strip()

    if isinstance(msg_field, str):
        try:
            msgs = json.loads(msg_field)
        except Exception:
            msgs = None
        if isinstance(msgs, list):
            if _has_duplicate_function_call([m for m in msgs if isinstance(m, dict)]):
                return {
                    "_skip": True,
                    "id": doc_id,
                    "metadata": {"dataset": "Toucan-1.5M", "raw": raw, "skip_reason": "duplicate_function_call"},
                }
            text = _format_messages([m for m in msgs if isinstance(m, dict)])
    elif isinstance(msg_field, list):
        if _has_duplicate_function_call([m for m in msg_field if isinstance(m, dict)]):
            return {
                "_skip": True,
                "id": doc_id,
                "metadata": {"dataset": "Toucan-1.5M", "raw": raw, "skip_reason": "duplicate_function_call"},
            }
        text = _format_messages([m for m in msg_field if isinstance(m, dict)])

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

    def _map_role(raw_role: str) -> str:
        r = (raw_role or "").strip().lower()
        if r in ("human", "user", "user_input", "human_input"):
            return "user"
        if r in ("assistant", "gpt", "model", "model_response", "ai"):
            return "assistant"
        if r in ("tool", "tool_output", "output", "observation", "tool_return", "execution_result", "text_observation"):
            return "tool"
        if r in ("system", "instruction", "rules"):
            return "system"
        if r in (
            "code_action",
            "tool_call",
            "function_call",
            "action",
            "code_action.description",
            "thought",
            "reasoning",
        ):
            return "assistant"
        return "assistant"

    def _wrap_content(raw_role: str, content: str) -> str:
        r = (raw_role or "").strip().lower()
        if r in ("code_action", "tool_call", "function_call", "action"):
            return f"<|tool_call|>\n{content}\n</|tool_call|>"
        if r in ("code_action.description", "thought", "reasoning"):
            return f"<think>\n{content}\n</think>"
        return content

    def _format_role_block(role: str, content: str) -> str:
        role_s = _map_role(role)
        return f"<|{role_s}|>\n{content}".strip() if role_s else content

    def _format_messages_with_roles(conv_list: list[dict]) -> str:
        parts: list[str] = []
        for m in conv_list:
            if not isinstance(m, dict):
                continue
            raw_role = _to_str(m.get("role") or m.get("from") or m.get("speaker") or m.get("name") or "").strip()
            content = _to_str(
                m.get("content") if "content" in m else (m.get("value") if "value" in m else m.get("text"))
            ).strip()
            if content:
                parts.append(_format_role_block(raw_role, _wrap_content(raw_role, content)) if raw_role else content)
        return "\n\n".join(parts).strip()

    def _format_steps_with_roles(steps_list: list[dict]) -> str:
        parts: list[str] = []
        for s in steps_list:
            if not isinstance(s, dict):
                continue
            raw_role = _to_str(s.get("source") or s.get("class_") or "").strip()
            main = _to_str(s.get("content")).strip()
            desc = _to_str(s.get("description")).strip()
            if raw_role.strip().lower() in ("code_action", "tool_call", "function_call", "action") and desc:
                desc_role = f"{raw_role}.description" if raw_role else "description"
                parts.append(_format_role_block(desc_role, _wrap_content(desc_role, desc)))
            if main:
                parts.append(_format_role_block(raw_role, _wrap_content(raw_role, main)) if raw_role else main)
            if not raw_role.strip().lower() in ("code_action", "tool_call", "function_call", "action") and desc:
                desc_role = f"{raw_role}.description" if raw_role else "description"
                parts.append(_format_role_block(desc_role, _wrap_content(desc_role, desc)))
        return "\n\n".join(parts).strip()

    conv = raw.get("conversations")
    if isinstance(conv, list):
        text = _format_messages_with_roles([m for m in conv if isinstance(m, dict)])

    # Nemotron-style fallback: some datasets use top-level `messages` with {role, content}
    if not text:
        msgs = raw.get("messages")
        if isinstance(msgs, list):
            text = _format_messages_with_roles([m for m in msgs if isinstance(m, dict)])

    # db-style fallback: top-level `content` is a list of step dicts containing `content`/`description`
    if not text:
        steps = raw.get("content")
        if isinstance(steps, list):
            text = _format_steps_with_roles([s for s in steps if isinstance(s, dict)])

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
    if text:
        stripped = text.strip()
        if stripped.startswith("Task:") and "\nAction:" in stripped and "Agent." in stripped:
            task_part, action_part = stripped.split("\nAction:", 1)
            task_part = task_part.strip()
            action_part = action_part.strip()
            text = f"<|user|>\n{task_part}\n\n<|assistant|>\nAction:\n{action_part}".strip()
        else:
            # Normalize simple user/assistant chat logs inside Hephaestus text
            lines = text.splitlines()
            role_re = re.compile(r"^(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
            blocks: list[tuple[str, list[str]]] = []
            current_role = None
            for ln in lines:
                m = role_re.match(ln.strip())
                if m:
                    role = m.group(1).lower()
                    content = m.group(2)
                    if current_role:
                        blocks.append((current_role, current_lines))
                    current_role = role
                    current_lines = [content] if content else []
                else:
                    if current_role:
                        current_lines.append(ln)
            if current_role:
                blocks.append((current_role, current_lines))
            if blocks:
                parts = []
                for role, content_lines in blocks:
                    content = "\n".join([c for c in content_lines if c is not None]).strip()
                    parts.append(f"<|{role}|>\n{content}".strip() if content else f"<|{role}|>")
                text = "\n\n".join(parts).strip()
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
            if role_s == "assistant":
                rc = m.get("reasoning_content")
                if isinstance(rc, str) and rc.strip():
                    parts.append(rc.strip())
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
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


