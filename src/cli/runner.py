from __future__ import annotations

"""CLI entrypoint (single).

Run:
  PYTHONPATH=src python -m cli.runner ...
"""

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="UData: run the Datatrove-backed pipeline")
    ap.add_argument("--datasets-root", default=str(Path.home() / "workspace" / "datasets"))
    ap.add_argument("--config-dir", default="configs/datasets")
    ap.add_argument("--out-root", default="out/datatrove_documents")
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
        "--prepare",
        action="store_true",
        help="Also write a CPT-friendly view under <out-root>/prepare/<dataset>/ with JSONL lines containing only {uuid,text}.",
    )
    ap.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "none"],
        help="Output compression. Default: none (plain .jsonl). Use 'gzip' to write .jsonl.gz.",
    )
    args = ap.parse_args(argv)

    # Import lazily so `--help` works even if datatrove isn't importable yet.
    from integrations.datatrove_backend import run_pipeline

    return run_pipeline(
        datasets_root=args.datasets_root,
        config_dir=args.config_dir,
        out_root=args.out_root,
        tasks_override=args.tasks,
        workers_override=args.workers,
        limit=args.limit,
        only=args.only,
        compression=(None if args.compression == "none" else args.compression),
        prepare=bool(args.prepare),
    )


if __name__ == "__main__":
    raise SystemExit(main())


