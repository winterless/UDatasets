from __future__ import annotations

"""CLI entrypoint (single).

Run:
  PYTHONPATH=src python -m cli.runner ...
"""

import argparse
import json
import os
from pathlib import Path
import shutil


def _slug(s: str) -> str:
    s = s.replace("/", "__").replace("\\", "__").replace("..", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)[:180]


def _ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.as_posix(), dst.as_posix())
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _shard_jsonl_file(src: Path, dst_dir: Path, *, target_bytes: int, prefix: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    part_idx = 0
    cur = 0
    out_fp = dst_dir / f"{prefix}.part{part_idx:05d}.jsonl"
    out_f = out_fp.open("wb")
    try:
        with src.open("rb") as f:
            for line in f:
                if not line:
                    continue
                out_f.write(line)
                cur += len(line)
                if cur >= target_bytes:
                    out_f.close()
                    part_idx += 1
                    out_fp = dst_dir / f"{prefix}.part{part_idx:05d}.jsonl"
                    out_f = out_fp.open("wb")
                    cur = 0
    finally:
        try:
            out_f.close()
        except Exception:
            pass


def prepare_sharded_configs(
    *,
    datasets_root: str,
    config_dir: str,
    out_root: str,
    shard_jsonl_mb: float,
    only: str,
    force: bool,
) -> str:
    """
    Create a derived config directory under <out_root>/_configs_sharded/ where JSONL datasets
    are redirected to a sharded view under <out_root>/_shards/<dataset>/.

    This is intentionally done in the CLI layer (not in datatrove) to keep changes minimal and robust.
    """
    shard_jsonl_mb = float(shard_jsonl_mb)
    if shard_jsonl_mb <= 0:
        return config_dir

    target_bytes = max(int(shard_jsonl_mb * 1024 * 1024), 256 * 1024)
    # Use absolute paths in derived configs, because downstream `pick_source()` joins data_dir with datasets_root.
    # If we wrote a relative data_dir like "out/_shards/...", it would incorrectly become <datasets_root>/out/_shards/...
    out_root_p = Path(out_root).expanduser().resolve()
    derived = out_root_p / "_configs_sharded"
    if force:
        shutil.rmtree(derived, ignore_errors=True)
    derived.mkdir(parents=True, exist_ok=True)

    datasets_root_p = Path(datasets_root).expanduser().resolve()
    cfg_dir_p = Path(config_dir).expanduser().resolve()

    for cfg_path in sorted(cfg_dir_p.glob("*.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        ds = cfg.get("dataset", {}) or {}
        name = (ds.get("name") or "").strip()
        if not name:
            continue
        if only and name != only:
            # copy as-is for non-targeted datasets (so the derived dir is complete)
            (derived / cfg_path.name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        sources = cfg.get("sources", []) or []
        if not sources or not isinstance(sources, list):
            (derived / cfg_path.name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        # Only shard the first usable jsonl source (simple + matches our configs).
        s0 = sources[0] if isinstance(sources[0], dict) else {}
        if (s0.get("type") or "").strip() != "jsonl":
            (derived / cfg_path.name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        data_dir = str(s0.get("data_dir") or "").strip()
        glob_pat = str(s0.get("glob") or "").strip()
        base = datasets_root_p / data_dir
        files = [p for p in base.glob(glob_pat) if p.is_file()] if (base.exists() and glob_pat) else []

        shard_dir = out_root_p / "_shards" / name
        if force:
            shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir(parents=True, exist_ok=True)

        for p in files:
            # Always surface every input file into shard_dir:
            # - large .jsonl => split into multiple parts
            # - small/other  => link/copy into shard_dir
            rel = ""
            try:
                rel = p.relative_to(base).as_posix()
            except Exception:
                rel = p.name
            if rel.lower().endswith(".jsonl"):
                rel = rel[:-5]
            prefix = _slug(rel)
            if p.suffix.lower() == ".jsonl":
                try:
                    if p.stat().st_size >= target_bytes:
                        existing = list(shard_dir.glob(f"{prefix}.part*.jsonl"))
                        if not existing:
                            _shard_jsonl_file(p, shard_dir, target_bytes=target_bytes, prefix=prefix)
                        continue
                except Exception:
                    pass
            # fallback: expose as a single file
            _ensure_link_or_copy(p, shard_dir / f"{prefix}.jsonl")

        # rewrite source to point to shard_dir (absolute path is OK)
        cfg2 = dict(cfg)
        cfg2_sources = [dict(s0)]
        cfg2_sources[0]["data_dir"] = shard_dir.as_posix()
        cfg2_sources[0]["glob"] = "*.jsonl"
        cfg2["sources"] = cfg2_sources
        (derived / cfg_path.name).write_text(json.dumps(cfg2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return derived.as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="UData: run the Datatrove-backed pipeline")
    ap.add_argument("--datasets-root", default=str(Path.home() / "workspace" / "datasets"))
    ap.add_argument("--config-dir", default="configs/datasets")
    ap.add_argument("--out-root", default="out/datatrove_documents")
    ap.add_argument(
        "-j",
        "--parallelism",
        type=int,
        default=None,
        help="Convenience flag to set both --tasks and --workers to the same value. "
        "You can still override them individually with --tasks/--workers.",
    )
    ap.add_argument(
        "-J",
        "--dataset-parallelism",
        type=int,
        default=1,
        help="Run multiple datasets (configs) concurrently. Default: 1 (sequential). "
        "Note: each dataset still uses its own tasks/workers; keep -j small to avoid oversubscribing CPU/disk.",
    )
    ap.add_argument(
        "--shard-jsonl-mb",
        type=float,
        default=0.0,
        help="Pre-shard large JSONL input files into smaller *.jsonl shards (~MB) under <out-root>/_shards/<dataset>/ "
        "before running datatrove. This improves parallelism and avoids per-task token-budget truncating a single huge file. "
        "0 disables.",
    )
    ap.add_argument(
        "--system-ratio",
        type=float,
        default=0.0,
        help="With probability in [0,1], prepend raw['system'] (tool/spec instructions) into text for datasets that have it. "
        "Selection is stable based on doc id and --seed. 0 disables.",
    )
    ap.add_argument(
        "--system-max-chars",
        type=int,
        default=2000,
        help="Max chars of raw['system'] to prepend when --system-ratio>0 (to avoid huge repeated prompts).",
    )
    ap.add_argument(
        "--mixed",
        default="",
        help="Enable explicit mixed mode: run all datasets as inputs but write a single combined JSONL under "
        "<out-root>/mixed/<NAME>/. Output lines keep prepare schema {uuid,text} and prefix uuid with '<dataset>::' "
        "to avoid collisions.",
    )
    ap.add_argument(
        "--exclude-ids-dir",
        default="",
        help="Optional: folder containing prior outputs (or any files) from which to collect id/uuid values as a blacklist. "
        "Any new documents whose id/uuid is in that blacklist will be skipped. Works with prepare-only and mixed mode.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-run a dataset by deleting its <out-root>/_logs/<dataset>/ state and existing outputs for that dataset.",
    )
    ap.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Override number of datatrove tasks. If omitted, use config `executor.tasks` (fallback 20).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of datatrove workers. If omitted, use config `executor.workers` (fallback 8).",
    )
    ap.add_argument("--limit", type=int, default=-1, help="Limit documents (debug). -1 for full")
    ap.add_argument("--only", default="", help="Only run a single dataset by name (dataset.name)")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for stable sampling/shuffling (when enabled in dataset configs) and for --system-ratio selection.",
    )
    ap.add_argument(
        "--prepare",
        action="store_true",
        help="Also write a CPT-friendly view under <out-root>/prepare/<dataset>/ with JSONL lines containing only {uuid,text}.",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the CPT-friendly prepare output (uuid+text) and skip the full Document output. Implies --prepare.",
    )
    ap.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "none"],
        help="Output compression. Default: none (plain .jsonl). Use 'gzip' to write .jsonl.gz.",
    )
    args = ap.parse_args(argv)

    if args.parallelism is not None:
        # Default to setting both when not explicitly set.
        if args.tasks is None:
            args.tasks = int(args.parallelism)
        if args.workers is None:
            args.workers = int(args.parallelism)

    # Import lazily so `--help` works even if datatrove isn't importable yet.
    from integrations.datatrove_backend import run_pipeline

    datasets_root = Path(args.datasets_root).expanduser().resolve().as_posix()
    config_dir_in = Path(args.config_dir).expanduser().resolve().as_posix()
    out_root = Path(args.out_root).expanduser().resolve().as_posix()

    config_dir = prepare_sharded_configs(
        datasets_root=datasets_root,
        config_dir=config_dir_in,
        out_root=out_root,
        shard_jsonl_mb=float(args.shard_jsonl_mb),
        only=str(args.only or ""),
        force=bool(args.force),
    )

    return run_pipeline(
        datasets_root=datasets_root,
        config_dir=config_dir,
        out_root=out_root,
        tasks_override=args.tasks,
        workers_override=args.workers,
        limit=args.limit,
        only=args.only,
        compression=(None if args.compression == "none" else args.compression),
        prepare=bool(args.prepare or args.prepare_only),
        prepare_only=bool(args.prepare_only),
        dataset_parallelism=int(args.dataset_parallelism),
        force=bool(args.force),
        seed=int(args.seed),
        system_ratio=float(args.system_ratio),
        system_max_chars=int(args.system_max_chars),
        mixed_name=str(args.mixed or "").strip(),
        exclude_ids_dir=(str(Path(args.exclude_ids_dir).expanduser().resolve()) if args.exclude_ids_dir else ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())


