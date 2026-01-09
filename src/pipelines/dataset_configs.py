from __future__ import annotations

"""
Config helpers (NO datatrove imports).

This module owns reading dataset config JSONs and choosing a usable source based
on filesystem presence and optional dependency availability.
"""

import json
from pathlib import Path


def load_configs(config_dir: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(config_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_config_path"] = str(p)
        out.append(cfg)
    return out


def pick_source(cfg: dict, datasets_root: Path) -> dict | None:
    sources = cfg.get("sources", [])
    for s in sources:
        stype = (s.get("type") or "").strip()
        data_dir = (s.get("data_dir") or "").strip()
        glob = (s.get("glob") or "").strip()
        if not stype or not data_dir or not glob:
            continue
        base = datasets_root / data_dir
        if not base.exists():
            continue
        if not list(base.glob(glob)):
            continue
        # Keep optional deps checks here (still no datatrove import).
        if stype == "parquet":
            try:
                import pyarrow  # noqa: F401
            except Exception:
                continue
        return s
    return None


