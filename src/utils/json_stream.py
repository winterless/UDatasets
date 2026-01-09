from __future__ import annotations

"""
Pure-stdlib JSON streaming helpers (NO datatrove imports).
"""

import json


def iter_json_array_stream(f, *, buffer_size: int = 1 << 20):
    """
    Incrementally parse a top-level JSON array from a text file-like object.

    Yields decoded elements one-by-one.
    """
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0

    def fill() -> bool:
        nonlocal buf, pos
        if pos > 0:
            buf = buf[pos:]
            pos = 0
        chunk = f.read(buffer_size)
        if chunk:
            buf += chunk
        return bool(chunk)

    fill()

    while True:
        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        if pos < len(buf):
            break
        if not fill():
            return

    if pos >= len(buf) or buf[pos] != "[":
        raise ValueError("JSON array reader expected '[' at start.")
    pos += 1

    while True:
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf) and buf[pos] == ",":
                pos += 1
                continue
            break

        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        if pos < len(buf) and buf[pos] == "]":
            return

        while True:
            # Important: after refilling we may end up with leading whitespace at the new
            # buffer start (pos reset to 0). JSONDecoder.raw_decode does NOT skip it.
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                # Also handle the case where we refilled right before a comma separator.
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                break
            try:
                obj, end = decoder.raw_decode(buf, pos)
                pos = end
                yield obj
                break
            except json.JSONDecodeError:
                if not fill():
                    raise


