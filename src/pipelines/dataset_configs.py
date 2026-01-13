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
        globs = s.get("globs") or []
        if not isinstance(globs, list):
            globs = []
        globs = [str(g).strip() for g in globs if str(g).strip()]

        if not stype or not data_dir or (not glob and not globs):
            continue
        base = datasets_root / data_dir
        if not base.exists():
            continue
        if glob:
            if not list(base.glob(glob)):
                continue
        else:
            # union globs
            found_any = False
            for g in globs:
                if list(base.glob(g)):
                    found_any = True
                    break
            if not found_any:
                continue
        # Keep optional deps checks here (still no datatrove import).
        if stype == "parquet":
            try:
                import pyarrow  # noqa: F401
            except Exception:
                continue
        return s
    return None


