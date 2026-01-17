from __future__ import annotations

"""
Datatrove backend (single-file).

Goal: keep the interaction surface with datatrove as small as possible.
- Input: --datasets-root + --config-dir (configs/datasets/*.json)
- Output: --out-root/<dataset>/<rank>.jsonl.gz (Document dict per line)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import random
import time
import traceback
import threading
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, IO, Literal

try:
    import orjson  # type: ignore

    _HAS_ORJSON = True
except Exception:
    orjson = None  # type: ignore
    _HAS_ORJSON = False


# -------------------------
# Datatrove import boundary
# -------------------------


def ensure_datatrove_importable() -> None:
    """
    Make datatrove importable without modifying the datatrove repo.

    Priority:
    - If datatrove is already importable, do nothing.
    - Else, prepend DATATROVE_SRC (default: ~/workspace/datatrove/src) to sys.path.
    """
    try:
        import datatrove  # noqa: F401  # type: ignore[import-not-found]

        return
    except Exception:
        pass

    default_src = str(Path.home() / "workspace" / "datatrove" / "src")
    dt_src = os.environ.get("DATATROVE_SRC", default_src)
    if dt_src and dt_src not in sys.path:
        sys.path.insert(0, dt_src)


ensure_datatrove_importable()

from datatrove.executor import LocalPipelineExecutor  # noqa: E402  # type: ignore[import-not-found]
from datatrove.io import DataFileLike, DataFolderLike  # noqa: E402  # type: ignore[import-not-found]
from datatrove.pipeline.readers import CSVReader as CsvReader  # noqa: E402  # type: ignore[import-not-found]
from datatrove.pipeline.readers import ParquetReader  # noqa: E402  # type: ignore[import-not-found]
from datatrove.pipeline.readers.base import BaseDiskReader  # noqa: E402  # type: ignore[import-not-found]
from datatrove.pipeline.writers.disk_base import DiskWriter  # noqa: E402  # type: ignore[import-not-found]
from datatrove.pipeline.base import PipelineStep  # noqa: E402  # type: ignore[import-not-found]
from datatrove.utils.logging import logger  # noqa: E402  # type: ignore[import-not-found]

from pipelines.dataset_adapters import BASE_ADAPTERS  # noqa: E402
from pipelines.dataset_configs import load_configs, pick_source  # noqa: E402
from utils.json_stream import iter_json_array_stream  # noqa: E402


# -------------------------
# Exclude-ids helpers (module-level so they can be pickled for ProcessPoolExecutor)
# -------------------------


def _exclude_ids_extract_json_string_field(line: str, key: str, scan_limit: int) -> str | None:
    """
    Fast-path extractor for JSONL lines that are dict-like JSON objects.

    We only scan the head (scan_limit) to avoid spending time scanning huge "text" tails.
    Supports basic JSON escapes inside the string value.
    """
    if not line or line[0] != "{":
        return None
    needle = f"\"{key}\""
    head = line if len(line) <= scan_limit else line[:scan_limit]
    i = head.find(needle)
    if i < 0:
        return None
    i += len(needle)
    n = len(line)
    while i < n and line[i].isspace():
        i += 1
    if i >= n or line[i] != ":":
        return None
    i += 1
    while i < n and line[i].isspace():
        i += 1
    if i >= n or line[i] != "\"":
        return None
    i += 1
    out_chars: list[str] = []
    while i < n:
        c = line[i]
        if c == "\"":
            return "".join(out_chars)
        if c == "\\":
            i += 1
            if i >= n:
                return None
            esc = line[i]
            if esc in ("\\", "\"", "/"):
                out_chars.append(esc)
            elif esc == "b":
                out_chars.append("\b")
            elif esc == "f":
                out_chars.append("\f")
            elif esc == "n":
                out_chars.append("\n")
            elif esc == "r":
                out_chars.append("\r")
            elif esc == "t":
                out_chars.append("\t")
            elif esc == "u":
                if i + 4 >= n:
                    return None
                hexs = line[i + 1 : i + 5]
                try:
                    out_chars.append(chr(int(hexs, 16)))
                except Exception:
                    return None
                i += 4
            else:
                return None
        else:
            out_chars.append(c)
        i += 1
    return None


def _exclude_ids_scan_jsonl_range_to_file(
    path: str, start: int, end: int, scan_limit: int, out_path: str
) -> tuple[int, int]:
    """
    Scan a byte-range of a JSONL file and write extracted ids to `out_path` (one per line).
    Returns (lines_seen, ids_written).
    """
    out: set[str] = set()
    lines_seen = 0
    try:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, int(start)))
            if start > 0:
                f.readline()
            while True:
                pos = f.tell()
                if pos >= end:
                    break
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                lines_seen += 1
                if not s.startswith("{"):
                    out.add(s)
                    continue
                uid = (
                    _exclude_ids_extract_json_string_field(s, "uuid", scan_limit)
                    or _exclude_ids_extract_json_string_field(s, "id", scan_limit)
                    or _exclude_ids_extract_json_string_field(s, "doc_id", scan_limit)
                )
                if uid:
                    out.add(uid)
    except Exception:
        out = set()
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wt", encoding="utf-8") as w:
            for s in out:
                w.write(s)
                w.write("\n")
    except Exception:
        return (lines_seen, 0)
    return (lines_seen, len(out))


# -------------------------
# Global token pool sampling (pre-shard by token budget across all input files)
# -------------------------


def _global_pool_sample_jsonl_to_shards(
    *,
    input_files: list[Path],
    base_adapter: Callable,
    dataset_name: str,
    out_dir: Path,
    tasks: int,
    token_budget: int,
    seed: int,
    chars_per_token: float,
    force: bool,
) -> Path:
    """
    Build a "global pool" sample across ALL input files, then split into `tasks` shard files.

    Behavior:
    - Deterministic order: shuffle input_files by seed, then stream lines until token_budget reached.
    - Uses adapter-produced `text` length to estimate tokens (same heuristic as TokenBudgetLimiter).
    - Writes raw JSON lines into `out_dir/pool_${rank}.jsonl`, so datatrove can re-read and adapt normally.
    """
    tasks = max(1, int(tasks))
    token_budget = max(1, int(token_budget))
    chars_per_token = float(chars_per_token)
    chars_per_token = 4.0 if chars_per_token <= 0 else chars_per_token

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "_DONE"
    shard_paths = [out_dir / f"pool_{r:05d}.jsonl" for r in range(tasks)]

    if marker.exists() and all(p.exists() and p.stat().st_size > 0 for p in shard_paths) and not force:
        print(f"[info] {dataset_name}: global_pool reuse {out_dir} (tasks={tasks})", file=sys.stderr)
        return out_dir

    if force:
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [out_dir / f"pool_{r:05d}.jsonl" for r in range(tasks)]

    # ensure clean slate
    for p in shard_paths:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        marker.unlink()
    except Exception:
        pass

    rng = random.Random(int(seed))
    files = list(input_files)
    rng.shuffle(files)

    if not files:
        print(f"[warn] {dataset_name}: global_pool has 0 input files; skipping", file=sys.stderr)
        for p in shard_paths:
            p.write_text("", encoding="utf-8")
        marker.write_text(json.dumps({"dataset": dataset_name, "tasks": tasks, "token_budget": token_budget, "docs": 0}) + "\n")
        return out_dir

    outs = [p.open("wb") for p in shard_paths]

    def _est_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / max(chars_per_token, 1e-6)))

    t0 = time.monotonic()
    last_log = t0
    total_tok = 0
    total_docs = 0
    total_bad = 0
    rr = 0
    files_done = 0

    try:
        for fp in files:
            files_done += 1
            try:
                with fp.open("rb") as f:
                    for li, raw_line in enumerate(f):
                        if total_tok >= token_budget:
                            break
                        b = raw_line.strip()
                        if not b:
                            continue
                        try:
                            obj = json.loads(b.decode("utf-8", errors="replace"))
                        except Exception:
                            total_bad += 1
                            continue
                        if not isinstance(obj, dict):
                            obj = {"value": obj}
                        try:
                            doc = base_adapter(None, obj, fp.as_posix(), li)  # type: ignore[arg-type]
                        except Exception:
                            total_bad += 1
                            continue
                        if not isinstance(doc, dict):
                            total_bad += 1
                            continue
                        text = doc.get("text")
                        if not isinstance(text, str) or not text.strip():
                            total_bad += 1
                            continue
                        t = _est_tokens(text)
                        if t <= 0:
                            continue
                        # write original line (raw record) to a shard
                        r = rr % tasks
                        rr += 1
                        outs[r].write(raw_line if raw_line.endswith(b"\n") else (raw_line + b"\n"))
                        total_tok += t
                        total_docs += 1

                        now = time.monotonic()
                        if now - last_log >= 5.0:
                            dt = max(now - t0, 1e-6)
                            print(
                                f"[info] {dataset_name}: global_pool building... "
                                f"files={files_done}/{len(files)} docs={total_docs:,} bad={total_bad:,} "
                                f"tok≈{total_tok:,}/{token_budget:,} rate≈{(total_docs/dt):,.1f} docs/s",
                                file=sys.stderr,
                            )
                            last_log = now
                    if total_tok >= token_budget:
                        break
            except Exception:
                total_bad += 1
                continue

        dt = max(time.monotonic() - t0, 1e-6)
        print(
            f"[info] {dataset_name}: global_pool built shards={tasks} docs={total_docs:,} bad={total_bad:,} "
            f"tok≈{total_tok:,}/{token_budget:,} in {dt:.1f}s",
            file=sys.stderr,
        )
    finally:
        for h in outs:
            try:
                h.close()
            except Exception:
                pass

    marker.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "tasks": tasks,
                "token_budget": token_budget,
                "seed": int(seed),
                "chars_per_token": float(chars_per_token),
                "docs": int(total_docs),
                "tokens_est": int(total_tok),
                "bad": int(total_bad),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def _merge_jsonl_folder_to_n(
    folder: Path,
    *,
    target_files: int,
    keep_inputs: bool = False,
    prefix: str = "merged",
) -> list[Path]:
    """
    Merge many *.jsonl / *.jsonl.gz files under `folder` into `target_files` outputs.

    - Streaming concat, deterministic by filename sort.
    - Balances by size (greedy bin packing) so outputs are roughly even.
    - Writes to temp files then atomically renames.
    """
    folder = Path(folder)
    target_files = max(1, int(target_files))
    keep_inputs = bool(keep_inputs)
    if not folder.exists():
        return []

    # collect inputs
    inputs: list[Path] = []
    for pat in ("*.jsonl", "*.jsonl.gz"):
        inputs.extend(list(folder.glob(pat)))
    inputs = sorted({p for p in inputs if p.is_file()})
    if len(inputs) <= target_files:
        return inputs

    # group by extension so we don't mix .jsonl and .jsonl.gz
    by_ext: dict[str, list[Path]] = {}
    for p in inputs:
        ext = ".jsonl.gz" if p.name.endswith(".jsonl.gz") else ".jsonl"
        by_ext.setdefault(ext, []).append(p)

    written: list[Path] = []
    for ext, files in by_ext.items():
        if len(files) <= target_files:
            written.extend(files)
            continue

        sizes = []
        for p in files:
            try:
                sizes.append((p, p.stat().st_size))
            except Exception:
                sizes.append((p, 0))
        sizes.sort(key=lambda x: x[1], reverse=True)

        k = min(target_files, len(files))
        bins: list[list[Path]] = [[] for _ in range(k)]
        bin_sizes = [0] * k
        for p, sz in sizes:
            i = min(range(k), key=lambda j: bin_sizes[j])
            bins[i].append(p)
            bin_sizes[i] += int(sz)

        out_paths = [folder / f"{prefix}_{i:05d}{ext}" for i in range(k)]
        tmp_paths = [folder / f".{prefix}_{i:05d}{ext}.tmp" for i in range(k)]

        # write
        for i in range(k):
            tmp = tmp_paths[i]
            outp = out_paths[i]
            try:
                tmp.unlink()
            except Exception:
                pass
            with tmp.open("wb") as w:
                for src in sorted(bins[i]):
                    try:
                        with src.open("rb") as r:
                            shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
                    except Exception:
                        continue
            try:
                tmp.replace(outp)
            except Exception:
                # fallback rename
                try:
                    shutil.move(tmp.as_posix(), outp.as_posix())
                except Exception:
                    pass
            written.append(outp)

        if not keep_inputs:
            for p in files:
                # don't delete the new outputs (in case names overlap)
                if p in out_paths:
                    continue
                try:
                    p.unlink()
                except Exception:
                    pass

    return written


def _pool_worker_parquet_to_shards(
    *,
    worker_idx: int,
    files: list[str],
    adapter_kind: str,
    dataset_name: str,
    out_dir: str,
    tasks: int,
    token_budget: int,
    seed: int,
    chars_per_token: float,
    batch_size: int = 1024,
) -> dict:
    """
    Worker: read parquet files, run adapter to get text, stop at token_budget,
    and write sampled raw rows to per-rank jsonl shard files under out_dir.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as e:
        return {"worker": worker_idx, "docs": 0, "tokens": 0, "bad": 0, "error": f"pyarrow missing: {e}"}

    base_adapter = BASE_ADAPTERS.get(adapter_kind)
    if not base_adapter:
        return {"worker": worker_idx, "docs": 0, "tokens": 0, "bad": 0, "error": f"unknown adapter: {adapter_kind}"}

    tasks = max(1, int(tasks))
    token_budget = max(1, int(token_budget))
    chars_per_token = 4.0 if float(chars_per_token) <= 0 else float(chars_per_token)
    rng = random.Random(int(seed) + int(worker_idx) * 9973)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    outs = [open((outp / f"pool_{r:05d}.jsonl").as_posix(), "ab") for r in range(tasks)]

    def _est_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / max(chars_per_token, 1e-6)))

    docs = 0
    tokens = 0
    bad = 0
    rr = rng.randint(0, 1_000_000)

    try:
        for fp in files:
            if tokens >= token_budget:
                break
            try:
                pf = pq.ParquetFile(fp)
            except Exception:
                bad += 1
                continue

            row_idx = 0
            try:
                for batch in pf.iter_batches(batch_size=int(batch_size)):
                    if tokens >= token_budget:
                        break
                    try:
                        rows = batch.to_pylist()
                    except Exception:
                        bad += 1
                        continue
                    for row in rows:
                        if tokens >= token_budget:
                            break
                        row_idx += 1
                        if not isinstance(row, dict):
                            row = {"value": row}
                        try:
                            doc = base_adapter(None, row, fp, row_idx)  # type: ignore[arg-type]
                        except Exception:
                            bad += 1
                            continue
                        if not isinstance(doc, dict):
                            bad += 1
                            continue
                        text = doc.get("text")
                        if not isinstance(text, str) or not text.strip():
                            bad += 1
                            continue
                        t = _est_tokens(text)
                        if t <= 0:
                            continue
                        # write raw row (not doc) so datatrove can re-adapt normally
                        try:
                            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                        except Exception:
                            bad += 1
                            continue
                        r = rr % tasks
                        rr += 1
                        outs[r].write(line)
                        docs += 1
                        tokens += t
                        if tokens >= token_budget:
                            break
            except Exception:
                bad += 1
                continue
    finally:
        for h in outs:
            try:
                h.close()
            except Exception:
                pass

    return {"worker": worker_idx, "docs": int(docs), "tokens": int(tokens), "bad": int(bad), "dataset": dataset_name}


def _global_pool_sample_parquet_to_shards(
    *,
    input_files: list[Path],
    adapter_kind: str,
    dataset_name: str,
    out_dir: Path,
    tasks: int,
    token_budget: int,
    seed: int,
    chars_per_token: float,
    build_workers: int,
    force: bool,
) -> Path:
    """
    Parquet global pool (方案A): sample from parquet and write `tasks` JSONL shard files, then datatrove reads JSONL.

    Note: This parallelizes by splitting parquet files across processes and splitting token budget across workers.
    The total token budget is exact as "sum of per-worker budgets", but token estimation is still heuristic.
    """
    tasks = max(1, int(tasks))
    build_workers = max(1, int(build_workers))
    token_budget = max(1, int(token_budget))
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "_DONE"
    shard_paths = [out_dir / f"pool_{r:05d}.jsonl" for r in range(tasks)]

    if marker.exists() and all(p.exists() and p.stat().st_size > 0 for p in shard_paths) and not force:
        print(f"[info] {dataset_name}: global_pool(parquet) reuse {out_dir} (tasks={tasks})", file=sys.stderr)
        return out_dir

    if force:
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [out_dir / f"pool_{r:05d}.jsonl" for r in range(tasks)]

    for p in shard_paths:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        marker.unlink()
    except Exception:
        pass

    files = list(input_files)
    rng = random.Random(int(seed))
    rng.shuffle(files)
    if not files:
        print(f"[warn] {dataset_name}: global_pool(parquet) has 0 input files; skipping", file=sys.stderr)
        for p in shard_paths:
            p.write_text("", encoding="utf-8")
        marker.write_text(json.dumps({"dataset": dataset_name, "tasks": tasks, "token_budget": token_budget, "docs": 0}) + "\n")
        return out_dir

    # split budget exactly across workers
    w = min(build_workers, len(files))
    if build_workers > 1 and len(files) == 1:
        # Important UX note: our parallelism here is per-input-file, so a single huge parquet cannot be sped up
        # much by increasing build_workers (unless the parquet itself is pre-sharded into multiple files).
        print(
            f"[warn] {dataset_name}: global_pool(parquet) has only 1 input file; "
            f"pool_workers={int(build_workers)} won't help much. "
            f"Tip: split the parquet into many files to enable real parallel read/convert.",
            file=sys.stderr,
        )
    base = token_budget // w
    rem = token_budget % w
    budgets = [(base + 1) if i < rem else base for i in range(w)]

    # split files across workers (round-robin) for better balance
    file_lists: list[list[str]] = [[] for _ in range(w)]
    for i, fp in enumerate(files):
        file_lists[i % w].append(fp.as_posix())

    work_root = out_dir / "_work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print(
        f"[info] {dataset_name}: global_pool(parquet) building in parallel: pool_workers={w} tasks={tasks} token_budget={token_budget:,}",
        file=sys.stderr,
    )
    total_docs = 0
    total_bad = 0
    total_tok = 0
    with ProcessPoolExecutor(max_workers=w) as ex:
        futs = {}
        for wi in range(w):
            outw = (work_root / f"w{wi:03d}").as_posix()
            fut = ex.submit(
                _pool_worker_parquet_to_shards,
                worker_idx=wi,
                files=file_lists[wi],
                adapter_kind=adapter_kind,
                dataset_name=dataset_name,
                out_dir=outw,
                tasks=tasks,
                token_budget=budgets[wi],
                seed=int(seed),
                chars_per_token=float(chars_per_token),
            )
            futs[fut] = wi

        done = 0
        for fut in as_completed(futs):
            wi = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"worker": wi, "docs": 0, "tokens": 0, "bad": 0, "error": str(e)}
            total_docs += int(res.get("docs", 0) or 0)
            total_bad += int(res.get("bad", 0) or 0)
            total_tok += int(res.get("tokens", 0) or 0)
            done += 1
            print(
                f"[info] {dataset_name}: global_pool(parquet) worker {wi} done ({done}/{w}) "
                f"docs={int(res.get('docs',0)):,} tok≈{int(res.get('tokens',0)):,} bad={int(res.get('bad',0)):,}",
                file=sys.stderr,
            )

    # merge: concatenate per-worker shard files into final shard files (deterministic by worker index)
    for r in range(tasks):
        outp = shard_paths[r]
        with outp.open("wb") as out_f:
            for wi in range(w):
                part = work_root / f"w{wi:03d}" / f"pool_{r:05d}.jsonl"
                if not part.exists():
                    continue
                try:
                    with part.open("rb") as in_f:
                        shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
                except Exception:
                    continue

    keep_work = bool(os.environ.get("UDATA_GLOBAL_POOL_KEEP_WORK", "").strip())
    if not keep_work:
        shutil.rmtree(work_root, ignore_errors=True)

    dt = max(time.monotonic() - t0, 1e-6)
    print(
        f"[info] {dataset_name}: global_pool(parquet) built shards={tasks} docs={total_docs:,} bad={total_bad:,} "
        f"tok≈{total_tok:,}/{token_budget:,} in {dt:.1f}s",
        file=sys.stderr,
    )
    marker.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "tasks": tasks,
                "token_budget": token_budget,
                "seed": int(seed),
                "chars_per_token": float(chars_per_token),
                "pool_workers": int(w),
                "docs": int(total_docs),
                "tokens_est": int(total_tok),
                "bad": int(total_bad),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir

# -------------------------
# Minimal writer
# -------------------------


class StdJsonlWriter(DiskWriter):
    """
    JsonlWriter using stdlib json (avoids orjson dependency).
    """

    default_output_filename: str = "${rank}.jsonl"
    name = "📝 StdJsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str | None = None,
        compression: str | None = "gzip",
        adapter: Callable | None = None,
        expand_metadata: bool = False,
        max_file_size: int = -1,
    ):
        super().__init__(
            output_folder,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            expand_metadata=expand_metadata,
            # Use binary mode so we can enable `max_file_size` file rolling (datatrove requires wb).
            mode="wb",
            max_file_size=max_file_size,
        )

    def _write(self, document: dict, file_handler: IO, _filename: str):
        use_orjson = bool(os.environ.get("UDATA_USE_ORJSON", "1").strip()) and _HAS_ORJSON
        if use_orjson:
            try:
                file_handler.write(orjson.dumps(document, option=orjson.OPT_APPEND_NEWLINE))  # type: ignore[union-attr]
                return
            except Exception:
                # Fall back to stdlib json for objects orjson can't handle.
                pass
        line = (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")
        file_handler.write(line)


class TokenBudgetLimiter(PipelineStep):
    """
    Stop the pipeline after writing approximately `token_budget` tokens.

    Token estimation is intentionally rough:
      tokens ~= len(text) / chars_per_token (default 4).

    Note: this limiter is per-rank. For a true global budget, run with tasks=1.
    """

    name = "🎛️ TokenBudgetLimiter"

    def __init__(self, token_budget: int, *, seed: int = 42, chars_per_token: float = 4.0):
        super().__init__()
        self.token_budget = int(token_budget)
        self.chars_per_token = float(chars_per_token)
        self._rng = random.Random(int(seed))

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        # very rough heuristic; good enough for budgeted sampling
        return max(1, int(len(text) / max(self.chars_per_token, 1e-6)))

    def run(self, data, rank: int = 0, world_size: int = 1):  # noqa: ANN001
        consumed = 0
        yielded_any = False
        for doc in data:
            t = self._estimate_tokens(getattr(doc, "text", "") or "")
            # Ensure we yield at least one document if any exist, even if it exceeds the budget.
            if consumed + t > self.token_budget and yielded_any:
                print(
                    f"[rank {rank}/{world_size}] TokenBudgetLimiter: stop (consumed≈{consumed} >= budget={self.token_budget})",
                    file=sys.stderr,
                )
                break
            consumed += t
            yielded_any = True
            yield doc
            if consumed >= self.token_budget:
                print(
                    f"[rank {rank}/{world_size}] TokenBudgetLimiter: stop (consumed≈{consumed} >= budget={self.token_budget})",
                    file=sys.stderr,
                )
                break


class ProgressLogger(PipelineStep):
    """
    Periodically log progress from inside datatrove worker processes.

    This is intentionally lightweight and stderr-only. It helps diagnose where a run "hangs":
    - no progress logs => stuck before workers start (file discovery / planning)
    - progress logs but low rate => CPU-bound adapter or slow IO
    - progress logs stop suddenly => token budget hit or input exhausted
    """

    name = "📈 ProgressLogger"

    def __init__(
        self,
        *,
        dataset_name: str,
        every_docs: int = 20000,
        every_seconds: float = 30.0,
        chars_per_token: float = 4.0,
    ):
        super().__init__()
        self.dataset_name = str(dataset_name or "")
        self.every_docs = int(every_docs)
        self.every_seconds = float(every_seconds)
        self.chars_per_token = float(chars_per_token)

    def run(self, data, rank: int = 0, world_size: int = 1):  # noqa: ANN001
        if self.every_docs <= 0 and self.every_seconds <= 0:
            yield from data
            return

        t0 = time.monotonic()
        last_t = t0
        n = 0
        last_n = 0
        token_est = 0

        def _maybe_log(force: bool = False) -> None:
            nonlocal last_t, last_n
            now = time.monotonic()
            if not force:
                if self.every_docs > 0 and (n - last_n) >= self.every_docs:
                    pass
                elif self.every_seconds > 0 and (now - last_t) >= self.every_seconds:
                    pass
                else:
                    return

            dt = max(now - t0, 1e-6)
            dn = n - last_n
            dts = max(now - last_t, 1e-6)
            rate = dn / dts
            total_rate = n / dt
            tok = token_est
            tok_s = tok / dt if tok > 0 else 0.0
            print(
                f"[rank {rank}/{world_size}] {self.dataset_name}: docs={n:,} "
                f"rate={rate:,.1f}/s avg={total_rate:,.1f}/s"
                + (f" tok≈{tok:,} tok/s≈{tok_s:,.1f}" if tok > 0 else ""),
                file=sys.stderr,
            )
            last_t = now
            last_n = n

        # initial heartbeat so user knows ranks started
        _maybe_log(force=True)
        for doc in data:
            n += 1
            try:
                text = getattr(doc, "text", "") or ""
                if isinstance(text, str) and text:
                    token_est += max(1, int(len(text) / max(self.chars_per_token, 1e-6)))
            except Exception:
                pass
            _maybe_log(force=False)
            yield doc

        # final summary
        _maybe_log(force=True)


# -------------------------
# Extra reader: json top-level array
# -------------------------

class JsonArrayReader(BaseDiskReader):
    name = "📚 JsonArray"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.compression = compression

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                # Some repos store large JSON files via Git LFS. If LFS objects were not pulled,
                # the file on disk is a small pointer text file, not valid JSON.
                head = f.read(200)
                if head.startswith("version https://git-lfs.github.com/spec/v1"):
                    logger.warning(
                        f"Skipping `{filepath}`: looks like a Git LFS pointer file (run `git lfs pull` to fetch real content)."
                    )
                    return
                try:
                    f.seek(0)
                except Exception:
                    # If not seekable, we'll fall through and let the parser error; it's safer than buffering whole file.
                    pass

                for li, obj in enumerate(iter_json_array_stream(f)):
                    with self.track_time():
                        if not isinstance(obj, dict):
                            obj = {"value": obj}
                        document = self.get_document_from_dict(obj, filepath, li)
                        if not document:
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")
            except (ValueError, json.JSONDecodeError) as e:
                # ValueError: not a top-level JSON array (e.g. starts with '{' or random text)
                # JSONDecodeError: truncated/corrupted content
                logger.warning(f"Skipping `{filepath}`: failed to parse as top-level JSON array ({e})")


# -------------------------
# Extra reader: jsonl (safe)
# -------------------------
class SafeJsonlReader(BaseDiskReader):
    """
    A safer JSONL reader than datatrove's built-in JsonlReader for messy corpora.

    Why:
    - Some datasets contain blank lines, "null" lines, or partially-written/corrupted JSON.
    - Older datatrove JsonlReader paths may assume each parsed line is a dict and do `line.get(...)`,
      which crashes when json.loads(line) returns None.

    Behavior:
    - Skip blank lines, `null` lines, and invalid JSON lines (log warning).
    - If a JSON line parses but is not a dict (e.g. list/str/number), wrap it as {"value": ...}.
    """

    name = "🐿 SafeJsonl"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.compression = compression

    def read_file(self, filepath: str):
        bad_json = 0
        null_lines = 0
        use_orjson = bool(os.environ.get("UDATA_USE_ORJSON", "1").strip()) and _HAS_ORJSON

        # Fast path: binary + orjson (avoids stdlib json overhead on huge JSONL).
        if use_orjson:
            with self.data_folder.open(filepath, "rb", compression=self.compression) as f:
                try:
                    head = f.read(200)
                    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                        logger.warning(
                            f"Skipping `{filepath}`: looks like a Git LFS pointer file (run `git lfs pull` to fetch real content)."
                        )
                        return
                    try:
                        f.seek(0)
                    except Exception:
                        pass

                    for li, raw_line in enumerate(f):
                        b = raw_line.strip()
                        if not b:
                            continue
                        try:
                            obj = orjson.loads(b)  # type: ignore[union-attr]
                        except Exception:
                            bad_json += 1
                            continue
                        if obj is None:
                            null_lines += 1
                            continue
                        if not isinstance(obj, dict):
                            obj = {"value": obj}
                        with self.track_time():
                            document = self.get_document_from_dict(obj, filepath, li)
                            if not document:
                                continue
                        yield document
                except UnicodeDecodeError as e:
                    logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")
                finally:
                    if bad_json:
                        logger.warning(f"In `{filepath}`: skipped {bad_json} invalid JSONL lines")
                    if null_lines:
                        logger.warning(f"In `{filepath}`: skipped {null_lines} `null` JSONL lines")
            return

        # Fallback: text + stdlib json
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                head = f.read(200)
                if head.startswith("version https://git-lfs.github.com/spec/v1"):
                    logger.warning(
                        f"Skipping `{filepath}`: looks like a Git LFS pointer file (run `git lfs pull` to fetch real content)."
                    )
                    return
                try:
                    f.seek(0)
                except Exception:
                    pass

                for li, raw_line in enumerate(f):
                    s = raw_line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except (ValueError, json.JSONDecodeError):
                        bad_json += 1
                        continue
                    if obj is None:
                        null_lines += 1
                        continue
                    if not isinstance(obj, dict):
                        obj = {"value": obj}
                    with self.track_time():
                        document = self.get_document_from_dict(obj, filepath, li)
                        if not document:
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")
            finally:
                if bad_json:
                    logger.warning(f"In `{filepath}`: skipped {bad_json} invalid JSONL lines")
                if null_lines:
                    logger.warning(f"In `{filepath}`: skipped {null_lines} `null` JSONL lines")


# -------------------------
# Adapters: record -> Document
# -------------------------

# -------------------------
# Runner
# -------------------------

def make_adapter(
    base_adapter: Callable,
    dataset_name: str,
    config_path: str,
    *,
    system_ratio: float = 0.0,
    system_max_chars: int = 2000,
    seed: int = 42,
):
    def _adapter(self, data, path, id_in_file):
        doc = base_adapter(self, data, path, id_in_file)
        # Optionally mix in tool/spec instructions from raw["system"] for a subset of samples.
        # Selection is stable across parallelism/order via (seed, doc_id) hashing.
        if system_ratio and system_ratio > 0:
            try:
                import hashlib

                doc_id = str(doc.get("id") or "")
                md0 = doc.get("metadata") or {}
                raw0 = md0.get("raw") if isinstance(md0, dict) else None
                sys_text = raw0.get("system") if isinstance(raw0, dict) else None
                if isinstance(sys_text, str) and sys_text.strip() and doc_id:
                    ratio = float(system_ratio)
                    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
                    h = hashlib.sha1(f"{int(seed)}:{doc_id}".encode("utf-8")).digest()
                    u = int.from_bytes(h[:8], "big") / float(2**64)
                    if u < ratio:
                        maxc = int(system_max_chars)
                        maxc = 0 if maxc < 0 else maxc
                        sys_snip = sys_text.strip() if maxc == 0 else sys_text.strip()[:maxc]
                        prefix = f"SYSTEM:\n{sys_snip}\n\n"
                        t = doc.get("text")
                        if isinstance(t, str):
                            doc["text"] = prefix + t
            except Exception:
                pass
        md = doc.get("metadata") or {}
        md.setdefault("dataset", dataset_name)
        md.setdefault("_config", config_path)
        doc["metadata"] = md
        return doc

    return _adapter


def build_reader(source: dict, datasets_root: Path, adapter_fn: Callable, *, paths_file: str | None = None):
    stype = source["type"]
    base = datasets_root / source["data_dir"]
    glob = source.get("glob") or ""
    shuffle_files = bool(source.get("_shuffle_files", False))
    if stype == "jsonl":
        return SafeJsonlReader(
            str(base),
            paths_file=paths_file,
            glob_pattern=(glob or None),
            recursive=("**" in glob),
            adapter=adapter_fn,
            shuffle_files=shuffle_files,
        )
    if stype == "parquet":
        return ParquetReader(
            str(base),
            paths_file=paths_file,
            glob_pattern=(glob or None),
            recursive=("**" in glob),
            adapter=adapter_fn,
            shuffle_files=shuffle_files,
        )
    if stype == "csv":
        return CsvReader(
            str(base),
            paths_file=paths_file,
            glob_pattern=(glob or None),
            recursive=("**" in glob),
            adapter=adapter_fn,
            shuffle_files=shuffle_files,
        )
    if stype == "json_array":
        return JsonArrayReader(
            str(base),
            paths_file=paths_file,
            glob_pattern=(glob or None),
            recursive=("**" in glob),
            adapter=adapter_fn,
            shuffle_files=shuffle_files,
        )
    raise ValueError(f"Unknown source type: {stype}")


def run_pipeline(
    *,
    datasets_root: str | Path,
    config_dir: str | Path,
    out_root: str | Path,
    tasks_override: int | None = None,
    workers_override: int | None = None,
    limit: int = -1,
    only: str = "",
    compression: str | None = None,
    prepare: bool = False,
    prepare_only: bool = False,
    token_budget: int | None = None,
    token_budget_parallel: bool = False,
    dataset_parallelism: int = 1,
    force: bool = False,
    mixed_name: str = "",
    exclude_ids_dir: str = "",
    system_ratio: float = 0.0,
    system_max_chars: int = 2000,
    seed: int = 42,
    chars_per_token: float = 4.0,
) -> int:
    """
    Backend API (NO argparse): run configured dataset pipelines using datatrove.
    """
    datasets_root = Path(datasets_root)
    config_dir = Path(config_dir)
    out_root = Path(out_root)

    cfgs = load_configs(config_dir)
    if not cfgs:
        print(f"[error] no configs found under {config_dir}", file=sys.stderr)
        return 2

    def _load_blacklist_ids(folder: str, *, workers: int = 1) -> set[str]:
        """
        Load a blacklist of ids/uuids from any files under `folder`.

        Supported inputs:
        - JSONL / JSONL.GZ where each line is a JSON object containing `uuid` or `id` (or `doc_id`)
        - plain text where each non-empty line is treated as an id/uuid
        - JSON lines that are a bare string are treated as an id/uuid
        """
        if not folder:
            return set()
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            print(f"[warn] exclude_ids_dir={folder!r} does not exist or is not a directory; ignoring", file=sys.stderr)
            return set()

        t0 = time.monotonic()
        workers = max(1, int(workers))
        # Default to process-based parallelism so it shows up as true multi-process work in `top`
        # and can bypass the GIL for Python-heavy scanning.
        # Users can override via UDATA_EXCLUDE_IDS_PARALLEL_MODE=thread|process.
        parallel_mode = os.environ.get("UDATA_EXCLUDE_IDS_PARALLEL_MODE", "process").strip().lower()
        if parallel_mode not in ("thread", "process"):
            parallel_mode = "thread"
        # Safety: process pools are not reliable on Windows/spawn-only environments for nested multiprocessing
        # setups; fall back to threads there.
        if sys.platform.startswith("win"):
            parallel_mode = "thread"
        ids: set[str] = set()
        files: list[Path] = []
        for p in root.rglob("*"):
            if p.is_file():
                files.append(p)
        files = sorted(files)

        def _add(v) -> None:  # noqa: ANN001
            if v is None:
                return
            if isinstance(v, (int, float)):
                v = str(v)
            if not isinstance(v, str):
                return
            s = v.strip()
            if s:
                ids.add(s)

        scan_limit = int(os.environ.get("UDATA_EXCLUDE_IDS_SCAN_LIMIT", "16384"))
        scan_limit = 1024 if scan_limit < 1024 else scan_limit

        def _extract_json_string_field(line: str, key: str) -> str | None:
            """
            Fast-path extractor for JSONL lines that are dict-like JSON objects.

            Why:
            - Your blacklist files can be huge and include a large "text" field.
            - json.loads(line) forces parsing that large field and is very slow.
            - We only need uuid/id/doc_id, so we scan and parse only that JSON string value.

            This supports basic JSON escapes inside the string value.
            """
            if not line or line[0] != "{":
                return None
            # Find `"key"` token first (including quotes to reduce false positives).
            needle = f"\"{key}\""
            # Only scan the head to avoid spending time scanning huge "text" tails.
            head = line if len(line) <= scan_limit else line[:scan_limit]
            i = head.find(needle)
            if i < 0:
                return None
            i += len(needle)
            # Skip to colon.
            n = len(line)
            while i < n and line[i].isspace():
                i += 1
            if i >= n or line[i] != ":":
                return None
            i += 1
            while i < n and line[i].isspace():
                i += 1
            if i >= n or line[i] != "\"":
                return None
            i += 1
            # Parse JSON string value.
            out_chars: list[str] = []
            while i < n:
                c = line[i]
                if c == "\"":
                    return "".join(out_chars)
                if c == "\\":
                    i += 1
                    if i >= n:
                        return None
                    esc = line[i]
                    if esc in ("\\", "\"", "/"):
                        out_chars.append(esc)
                    elif esc == "b":
                        out_chars.append("\b")
                    elif esc == "f":
                        out_chars.append("\f")
                    elif esc == "n":
                        out_chars.append("\n")
                    elif esc == "r":
                        out_chars.append("\r")
                    elif esc == "t":
                        out_chars.append("\t")
                    elif esc == "u":
                        # Minimal \uXXXX support.
                        if i + 4 >= n:
                            return None
                        hexs = line[i + 1 : i + 5]
                        try:
                            out_chars.append(chr(int(hexs, 16)))
                        except Exception:
                            return None
                        i += 4
                    else:
                        # Unknown escape; treat as failure.
                        return None
                else:
                    out_chars.append(c)
                i += 1
            return None

        chunk_mb = int(os.environ.get("UDATA_EXCLUDE_IDS_CHUNK_MB", "512"))
        chunk_mb = 64 if chunk_mb < 64 else chunk_mb
        chunk_bytes = int(chunk_mb * 1024 * 1024)

        def _scan_jsonl_range(path: str, start: int, end: int) -> set[str]:
            out: set[str] = set()
            try:
                with open(path, "rt", encoding="utf-8", errors="replace") as f:
                    f.seek(max(0, int(start)))
                    if start > 0:
                        # align to next full line
                        f.readline()
                    while True:
                        pos = f.tell()
                        if pos >= end:
                            break
                        line = f.readline()
                        if not line:
                            break
                        s = line.strip()
                        if not s:
                            continue
                        if not s.startswith("{"):
                            out.add(s)
                            continue
                        uid = (
                            _extract_json_string_field(s, "uuid")
                            or _extract_json_string_field(s, "id")
                            or _extract_json_string_field(s, "doc_id")
                        )
                        if uid:
                            out.add(uid)
            except Exception:
                return out
            return out

        def _scan_text_file(path: str) -> set[str]:
            out: set[str] = set()
            try:
                with open(path, "rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = str(line).strip()
                        if s:
                            out.add(s)
            except Exception:
                return out
            return out

        def _scan_jsonl_gz(path: str) -> set[str]:
            out: set[str] = set()
            try:
                import gzip

                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        if not s.startswith("{"):
                            out.add(s)
                            continue
                        uid = (
                            _extract_json_string_field(s, "uuid")
                            or _extract_json_string_field(s, "id")
                            or _extract_json_string_field(s, "doc_id")
                        )
                        if uid:
                            out.add(uid)
            except Exception:
                return out
            return out

        # Build scan tasks (file-level and chunk-level)
        tasks: list[tuple[str, str, int, int]] = []
        total_bytes = 0
        for p in files:
            try:
                fsize = p.stat().st_size
            except Exception:
                fsize = 0

            if p.suffix == ".gz" and p.name.endswith(".jsonl.gz"):
                tasks.append(("jsonl_gz", p.as_posix(), 0, 0))
                continue

            if p.suffix == ".jsonl" and fsize > 0 and workers > 1 and fsize >= chunk_bytes:
                total_bytes += fsize
                n_parts = int((fsize + chunk_bytes - 1) // chunk_bytes)
                n_parts = max(1, min(workers, n_parts))
                part_size = int((fsize + n_parts - 1) // n_parts)
                for i in range(n_parts):
                    start = i * part_size
                    end = min(fsize, (i + 1) * part_size)
                    tasks.append(("jsonl_range", p.as_posix(), int(start), int(end)))
                print(
                    f"[info] exclude-ids: {p.name}: split into {n_parts} chunk(s) (~{(part_size/(1024**2)):.0f}MB each) "
                    f"workers={workers} scan_limit={scan_limit}B",
                    file=sys.stderr,
                )
                continue

            if p.suffix == ".jsonl":
                tasks.append(("jsonl_range", p.as_posix(), 0, max(int(fsize), 1 << 60)))
                continue

            # fallback: treat as plain text
            tasks.append(("text", p.as_posix(), 0, 0))

        if workers <= 1 or len(tasks) <= 1:
            # sequential mode (still uses fast extractor)
            for kind, path, start, end in tasks:
                if kind == "jsonl_gz":
                    ids |= _scan_jsonl_gz(path)
                elif kind == "jsonl_range":
                    ids |= _scan_jsonl_range(path, start, end)
                else:
                    ids |= _scan_text_file(path)
        else:
            print(
                f"[info] exclude-ids: parallel scan starting (mode={parallel_mode} tasks={len(tasks)} workers={workers} "
                f"chunk_mb={chunk_mb} scan_limit={scan_limit})",
                file=sys.stderr,
            )
            done = 0
            done_bytes = 0
            last_log_t = time.monotonic()
            if parallel_mode == "process":
                # Process mode avoids GIL limits; workers write ids to temp files to avoid huge IPC/pickling.
                import hashlib

                tmp_root = out_root / "_exclude_ids_cache" / hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:12]
                tmp_root.mkdir(parents=True, exist_ok=True)
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {}
                    for idx, (kind, path, start, end) in enumerate(tasks):
                        if kind == "jsonl_range":
                            outp = (tmp_root / f"{Path(path).name}.part{idx:05d}.ids.txt").as_posix()
                            fut = ex.submit(
                                _exclude_ids_scan_jsonl_range_to_file, path, int(start), int(end), int(scan_limit), outp
                            )
                            futs[fut] = ("jsonl_range_file", outp, max(0, int(end) - int(start)))
                        else:
                            # fall back to thread-like scan inside the main process for gz/text
                            # (process-safe chunking for gzip isn't supported; text files are typically small)
                            if kind == "jsonl_gz":
                                got = _scan_jsonl_gz(path)
                                if got:
                                    ids |= got
                            else:
                                got = _scan_text_file(path)
                                if got:
                                    ids |= got

                    for fut in as_completed(futs):
                        _k, outp, b = futs[fut]
                        try:
                            fut.result()
                        except Exception:
                            pass
                        # merge this part
                        try:
                            with open(outp, "rt", encoding="utf-8", errors="replace") as r:
                                for line in r:
                                    s = line.strip()
                                    if s:
                                        ids.add(s)
                        except Exception:
                            pass
                        done += 1
                        done_bytes += int(b or 0)
                        now = time.monotonic()
                        if now - last_log_t >= 5.0 or done == len(futs):
                            pct = (100.0 * done_bytes / total_bytes) if total_bytes else 0.0
                            print(
                                f"[info] exclude-ids: progress {done}/{len(futs)} tasks, ids={len(ids):,}"
                                + (f", ~{pct:.1f}% bytes" if total_bytes else ""),
                                file=sys.stderr,
                            )
                            last_log_t = now
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {}
                    for kind, path, start, end in tasks:
                        if kind == "jsonl_gz":
                            fut = ex.submit(_scan_jsonl_gz, path)
                            futs[fut] = (kind, path, 0)
                        elif kind == "jsonl_range":
                            fut = ex.submit(_scan_jsonl_range, path, start, end)
                            futs[fut] = (kind, path, max(0, int(end) - int(start)))
                        else:
                            fut = ex.submit(_scan_text_file, path)
                            futs[fut] = (kind, path, 0)

                    for fut in as_completed(futs):
                        kind, path, b = futs[fut]
                        try:
                            got = fut.result()
                        except Exception:
                            got = set()
                        if got:
                            ids |= got
                        done += 1
                        done_bytes += int(b or 0)
                        now = time.monotonic()
                        if now - last_log_t >= 5.0 or done == len(futs):
                            pct = (100.0 * done_bytes / total_bytes) if total_bytes else 0.0
                            print(
                                f"[info] exclude-ids: progress {done}/{len(futs)} tasks, ids={len(ids):,}"
                                + (f", ~{pct:.1f}% bytes" if total_bytes else ""),
                                file=sys.stderr,
                            )
                            last_log_t = now

        dt = time.monotonic() - t0
        print(f"[info] exclude-ids: loaded {len(ids):,} ids from {root} in {dt:.1f}s", file=sys.stderr)
        return ids

    blacklist_ids_cache: set[str] | None = None

    def _get_blacklist_ids() -> set[str]:
        nonlocal blacklist_ids_cache
        if blacklist_ids_cache is not None:
            return blacklist_ids_cache
        if not exclude_ids_dir:
            blacklist_ids_cache = set()
            return blacklist_ids_cache

        # Simple persistent cache: if present and not --force, never rescan exclude_ids_dir.
        # Cache path is keyed only by the resolved exclude_ids_dir path (no expensive validation).
        cache_key = hashlib.sha1(str(exclude_ids_dir).encode("utf-8")).hexdigest()[:16]
        # Persisted cache lives under the existing out/ scratch namespace used by exclude-ids scanning.
        # User-facing convention: treat this as "the exclude-ids cache", not a generic `_cache` folder.
        cache_dir = out_root / "_exclude_ids_cache" / "persist" / cache_key
        cache_file = cache_dir / "ids.txt"
        cache_meta = cache_dir / "meta.json"
        if cache_file.exists() and not force:
            ids: set[str] = set()
            t0 = time.monotonic()
            try:
                with cache_file.open("rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            ids.add(s)
                dt = time.monotonic() - t0
                print(
                    f"[info] exclude-ids: using cache {cache_file} (ids={len(ids):,}, {dt:.1f}s). "
                    f"Use --force to rebuild.",
                    file=sys.stderr,
                )
                blacklist_ids_cache = ids
                return blacklist_ids_cache
            except Exception:
                # Fall back to rebuild below.
                pass

        exclude_workers_env = os.environ.get("UDATA_EXCLUDE_IDS_WORKERS", "").strip()
        exclude_workers = 0
        exclude_workers_src = ""
        if exclude_workers_env:
            try:
                exclude_workers = int(exclude_workers_env)
                exclude_workers_src = "env:UDATA_EXCLUDE_IDS_WORKERS"
            except Exception:
                exclude_workers = 0
        if exclude_workers <= 0:
            # Follow user-level parallelism: `-j` sets workers_override.
            if workers_override and int(workers_override) > 0:
                exclude_workers = int(workers_override)
                exclude_workers_src = "cli:-j"
            else:
                # Default: do NOT tie exclude scan parallelism to `tasks` (users often set tasks=1 to get 1 output file).
                exclude_workers = min(32, (os.cpu_count() or 1))
                exclude_workers_src = "default:cpu"
        exclude_workers = max(1, int(exclude_workers))
        print(
            f"[info] exclude-ids: workers={exclude_workers} ({exclude_workers_src}) "
            f"(override with UDATA_EXCLUDE_IDS_WORKERS; mode={os.environ.get('UDATA_EXCLUDE_IDS_PARALLEL_MODE','process')})",
            file=sys.stderr,
        )

        ids = _load_blacklist_ids(exclude_ids_dir, workers=exclude_workers)
        blacklist_ids_cache = ids

        # Write cache for next run
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_dir / f".ids.txt.tmp.{os.getpid()}"
            with tmp.open("wt", encoding="utf-8") as w:
                for s in sorted(ids):
                    w.write(s)
                    w.write("\n")
            tmp.replace(cache_file)
            cache_meta.write_text(
                json.dumps(
                    {
                        "exclude_ids_dir": str(exclude_ids_dir),
                        "ids": int(len(ids)),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[info] exclude-ids: wrote cache {cache_file} (ids={len(ids):,})", file=sys.stderr)
        except Exception:
            pass
        return blacklist_ids_cache

    def _stage(msg: str, *, dataset: str = "") -> None:
        ds = f" {dataset}:" if dataset else ""
        ts = time.strftime("%H:%M:%S")
        print(f"[stage {ts}]{ds} {msg}", file=sys.stderr)

    class ExcludeIdsFilter(PipelineStep):
        """
        Drop documents whose id is present in `blacklist_ids`.

        In mixed mode, also matches dataset-prefixed ids of the form `<dataset>::<id>`.
        """

        name = "🚫 ExcludeIdsFilter"

        def __init__(self, ids: set[str], *, dataset_name: str = "", mixed: bool = False):
            super().__init__()
            self._ids = ids
            self._dataset_name = str(dataset_name or "")
            self._mixed = bool(mixed)

        def run(self, data, rank: int = 0, world_size: int = 1):  # noqa: ANN001
            if not self._ids:
                yield from data
                return
            prefix = f"{self._dataset_name}::" if (self._mixed and self._dataset_name) else ""
            for doc in data:
                uid = str(getattr(doc, "id", "") or "")
                if not uid:
                    yield doc
                    continue
                if uid in self._ids:
                    continue
                if prefix and f"{prefix}{uid}" in self._ids:
                    continue
                yield doc

    def _run_one(cfg: dict) -> int:
        """
        Run one dataset config end-to-end, with verbose stage logging and full tracebacks on errors.
        """
        name = "<unknown>"
        try:
            ds = cfg.get("dataset", {})
            name = (ds.get("name") or "").strip() or name
            if name == "<unknown>":
                return 0
            if only and name != only:
                return 0

            _stage(f"start (config={cfg.get('_config_path','')})", dataset=name)

            enabled = cfg.get("enabled", ds.get("enabled", True))
            if enabled is not True:
                print(f"[skip] {name}: disabled (enabled={enabled!r})", file=sys.stderr)
                return 0

            exec_cfg = cfg.get("executor") or {}

            # --- fast resume guard (skip before any expensive work, including exclude-ids scan) ---
            ds_out = out_root / name
            logs_dir = out_root / "_logs" / name

            want_full = (not prepare_only) and (not mixed_name)
            want_prepare = bool(prepare or mixed_name)

            def _has_any_jsonl(folder: Path, pattern: str = "") -> bool:
                if not folder.exists():
                    return False
                if pattern:
                    return any(folder.glob(pattern))
                return any(folder.glob("*.jsonl")) or any(folder.glob("*.jsonl.gz")) or any(folder.glob("*.jsonl.*"))

            outputs_ok = True
            if want_full and not _has_any_jsonl(ds_out):
                outputs_ok = False
            if want_prepare:
                if mixed_name:
                    mixed_dir = out_root / "mixed" / mixed_name
                    # In mixed mode, each dataset writes files named "<dataset>__${rank}.jsonl".
                    if not _has_any_jsonl(mixed_dir, f"{name}__*.jsonl"):
                        outputs_ok = False
                else:
                    prep_dir = out_root / "prepare" / name
                    if not _has_any_jsonl(prep_dir):
                        outputs_ok = False

            if outputs_ok and not force:
                print(f"[skip] {name}: outputs already exist; skipping (use --force to re-run)", file=sys.stderr)
                return 0

            # --- token budget (config-driven) ---
            def _cfg_token_budget() -> int | None:
                if token_budget is not None:
                    return int(token_budget)
                if "token_budget" in exec_cfg:
                    try:
                        return int(exec_cfg.get("token_budget"))
                    except Exception:
                        return None
                if "token_billions" in exec_cfg:
                    try:
                        b = float(exec_cfg.get("token_billions"))
                        return int(b * 1_000_000_000)
                    except Exception:
                        return None
                return None

            eff_token_budget = _cfg_token_budget()
            eff_token_budget_parallel = bool(exec_cfg.get("token_budget_parallel", token_budget_parallel))
            eff_seed = int(exec_cfg.get("seed", seed))
            eff_chars_per_token = float(exec_cfg.get("chars_per_token", chars_per_token))
            token_budget_mode = str(exec_cfg.get("token_budget_mode", exec_cfg.get("token_mode", "")) or "").strip().lower()
            global_pool_enabled = bool(exec_cfg.get("global_token_pool", False)) or (token_budget_mode == "global_pool")

            # --- adapter ---
            adapter_kind = (cfg.get("adapter", {}) or {}).get("kind", "")
            base_adapter = BASE_ADAPTERS.get(adapter_kind)
            if not base_adapter:
                print(
                    f"[skip] {name}: unknown adapter kind {adapter_kind!r} ({cfg.get('_config_path')})",
                    file=sys.stderr,
                )
                return 0
            adapter_fn = make_adapter(
                base_adapter,
                name,
                cfg.get("_config_path", ""),
                system_ratio=float(system_ratio),
                system_max_chars=int(system_max_chars),
                seed=int(eff_seed),
            )
            _stage(
                f"adapter={adapter_kind} system_ratio={float(system_ratio)} system_max_chars={int(system_max_chars)} seed={eff_seed}",
                dataset=name,
            )

            # --- source selection ---
            source = pick_source(cfg, datasets_root)
            if not source:
                print(f"[skip] {name}: no usable source found (paths missing or deps missing)", file=sys.stderr)
                return 0
            _stage(
                f"source picked type={source.get('type')} data_dir={source.get('data_dir')} "
                f"glob={source.get('glob','')} globs={len(source.get('globs') or [])}",
                dataset=name,
            )

            def _get_path_keywords(src: dict) -> list[str]:
                """
                Optional file/folder keyword filter:
                Only keep files whose *relative path under data_dir* contains any of these substrings.

                Config keys supported:
                - source.path_keywords: ["low", "part00"] (list)
                - source.path_keyword: "low" (single string)
                - source.include_keywords: ["..."] (alias)
                """
                kws = src.get("path_keywords", src.get("include_keywords", []))
                if isinstance(kws, str):
                    kws = [kws]
                if not isinstance(kws, list):
                    kws = []
                if isinstance(src.get("path_keyword"), str) and src.get("path_keyword").strip():
                    kws = list(kws) + [src.get("path_keyword")]
                out: list[str] = []
                for k in kws:
                    s = str(k).strip()
                    if s:
                        out.append(s)
                # de-dupe, stable
                seen = set()
                dedup: list[str] = []
                for k in out:
                    if k not in seen:
                        seen.add(k)
                        dedup.append(k)
                return dedup

            path_keywords = _get_path_keywords(source)

            def _kw_match(rel_posix: str) -> bool:
                if not path_keywords:
                    return True
                rp = str(rel_posix)
                return any(k in rp for k in path_keywords)

            if eff_token_budget:
                # enable file-level shuffling to make early-stop sampling less biased
                source = dict(source)
                source["_shuffle_files"] = True

            # --- build paths_file if needed (multi-glob or keyword filter) ---
            paths_file: str | None = None
            globs = source.get("globs") or []
            if not isinstance(globs, list):
                globs = []
            globs = [str(g).strip() for g in globs if str(g).strip()]

            if globs:
                base = datasets_root / source["data_dir"]
                rel_paths: list[str] = []
                for g in globs:
                    for p in base.glob(g):
                        if p.is_file():
                            rp = p.relative_to(base).as_posix()
                            if _kw_match(rp):
                                rel_paths.append(rp)
                rel_paths = sorted(set(rel_paths))
                if not rel_paths:
                    print(f"[skip] {name}: globs matched 0 files under {base}", file=sys.stderr)
                    return 0
                paths_file_path = out_root / "_paths" / name / "paths.txt"
                paths_file_path.parent.mkdir(parents=True, exist_ok=True)
                paths_file_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
                paths_file = str(paths_file_path)
                _stage(f"paths_file generated (globs union): {len(rel_paths)} file(s)", dataset=name)
            elif path_keywords:
                base = datasets_root / source["data_dir"]
                pattern = (source.get("glob") or "").strip()
                rel_paths: list[str] = []
                if pattern:
                    for p in base.glob(pattern):
                        if p.is_file():
                            rp = p.relative_to(base).as_posix()
                            if _kw_match(rp):
                                rel_paths.append(rp)
                rel_paths = sorted(set(rel_paths))
                if not rel_paths:
                    print(
                        f"[skip] {name}: keyword filter matched 0 files under {base} (keywords={path_keywords})",
                        file=sys.stderr,
                    )
                    return 0
                paths_file_path = out_root / "_paths" / name / "paths.txt"
                paths_file_path.parent.mkdir(parents=True, exist_ok=True)
                paths_file_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
                paths_file = str(paths_file_path)
                _stage(f"paths_file generated (keyword filter {path_keywords}): {len(rel_paths)} file(s)", dataset=name)

            # --- reader ---
            reader = build_reader(source, datasets_root, adapter_fn, paths_file=paths_file)
            if limit != -1:
                reader.limit = limit

            ds_out.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            # --- scan files for stats / planning ---
            base = datasets_root / source["data_dir"]
            pattern = (source.get("glob") or "").strip()
            globs_for_stats = source.get("globs") or []
            if not isinstance(globs_for_stats, list):
                globs_for_stats = []
            globs_for_stats = [str(g).strip() for g in globs_for_stats if str(g).strip()]

            t_scan0 = time.monotonic()
            if globs_for_stats:
                matched_files: list[Path] = []
                for g in globs_for_stats:
                    matched_files.extend([p for p in base.glob(g) if p.is_file()])
            else:
                matched_files = [p for p in base.glob(pattern) if p.is_file()] if pattern else []
            if path_keywords:
                matched_files = [p for p in matched_files if _kw_match(p.relative_to(base).as_posix())]

            file_count = len(matched_files)
            total_bytes = 0
            for p in matched_files:
                try:
                    total_bytes += p.stat().st_size
                except Exception:
                    pass
            scan_ms = int((time.monotonic() - t_scan0) * 1000)
            if scan_ms >= 1500:
                print(
                    f"[info] {name}: file discovery took {scan_ms}ms (files={file_count}, base={base}, pattern={pattern or globs_for_stats})",
                    file=sys.stderr,
                )

            def _estimate_jsonl_file_tokens(p: Path, *, sample_lines: int = 200) -> float | None:
                """
                Estimate tokens for a JSONL file by sampling a few lines and extrapolating.
                Used as a guardrail to avoid per-task token budget truncating a single huge file when tasks>1.
                """
                try:
                    size_bytes = p.stat().st_size
                except Exception:
                    return None
                if size_bytes <= 0:
                    return 0.0
                n = 0
                bytes_sum = 0
                text_chars_sum = 0
                try:
                    with p.open("rb") as f:
                        for li in range(int(sample_lines)):
                            line_b = f.readline()
                            if not line_b:
                                break
                            b = line_b.strip()
                            if not b:
                                continue
                            bytes_sum += len(line_b)
                            try:
                                obj = json.loads(b.decode("utf-8", errors="replace"))
                            except Exception:
                                continue
                            if not isinstance(obj, dict):
                                obj = {"value": obj}
                            try:
                                doc = base_adapter(None, obj, p.as_posix(), li)  # type: ignore[arg-type]
                            except Exception:
                                continue
                            t = doc.get("text") if isinstance(doc, dict) else ""
                            if isinstance(t, str):
                                text_chars_sum += len(t)
                            n += 1
                except Exception:
                    return None

                if n <= 0 or bytes_sum <= 0:
                    return None
                avg_bytes_per_line = bytes_sum / n
                est_lines = max(1.0, float(size_bytes) / max(avg_bytes_per_line, 1e-6))
                avg_text_chars = text_chars_sum / n
                est_chars = est_lines * avg_text_chars
                return est_chars / max(eff_chars_per_token, 1e-6)

            # --- output file rolling ---
            min_mb = float(exec_cfg.get("min_shard_mb", exec_cfg.get("min_mb", 2)))
            target_mb = float(exec_cfg.get("target_shard_mb", exec_cfg.get("target_mb", 2)))
            min_bytes = int(min_mb * 1024 * 1024)
            target_bytes = max(int(target_mb * 1024 * 1024), 1)
            writer_max_file_size = -1 if total_bytes < min_bytes else target_bytes

            # --- tasks/workers planning ---
            if tasks_override is not None:
                tasks = int(tasks_override)
            elif "tasks" in exec_cfg:
                tasks = int(exec_cfg.get("tasks", 20))
            else:
                if eff_token_budget and not eff_token_budget_parallel:
                    tasks = 1
                else:
                    if total_bytes < min_bytes or file_count <= 1:
                        tasks = 1
                    else:
                        default_tasks = int(exec_cfg.get("default_tasks", 20))
                        tasks = max(1, min(file_count, default_tasks))

            if (not global_pool_enabled) and file_count > 0 and tasks > file_count:
                print(f"[warn] {name}: tasks={tasks} > input_files={file_count}; capping tasks to {file_count}", file=sys.stderr)
                tasks = file_count

            if (not global_pool_enabled) and eff_token_budget and tasks > 1 and source.get("type") == "jsonl":
                per_rank_budget = max(1, (int(eff_token_budget) + tasks - 1) // tasks)
                est_max_file_tokens = 0.0
                for p in sorted(matched_files, key=lambda x: getattr(x.stat(), "st_size", 0), reverse=True)[:5]:
                    est = _estimate_jsonl_file_tokens(p)
                    if est is not None:
                        est_max_file_tokens = max(est_max_file_tokens, float(est))
                if est_max_file_tokens > 0 and per_rank_budget < est_max_file_tokens:
                    max_tasks_no_trunc = max(1, int(int(eff_token_budget) // int(max(est_max_file_tokens, 1))))
                    if max_tasks_no_trunc < tasks:
                        print(
                            f"[warn] {name}: token_budget split across tasks would truncate a large JSONL file "
                            f"(per_rank_budget≈{per_rank_budget}, est_max_file_tokens≈{int(est_max_file_tokens)}). "
                            f"Capping tasks {tasks} -> {max_tasks_no_trunc} to avoid truncation. "
                            f"(Tip: to keep higher parallelism, pre-shard the large JSONL into multiple files.)",
                            file=sys.stderr,
                        )
                        tasks = max_tasks_no_trunc

            # Datatrove parallelism:
            # - `tasks`: how many sub-tasks/shards exist (often maps to output files)
            # - `workers`: how many worker processes run those tasks in parallel
            #
            # Users often set `tasks` high (e.g. 192) expecting speedups; but if `workers` stays low (default=8),
            # execution is still limited to 8 concurrent processes. So we default workers to CPU count (capped),
            # unless explicitly overridden.
            cpu = int(os.cpu_count() or 1)
            if workers_override is not None:
                workers = int(workers_override)
                workers_src = "cli"
            elif "workers" in exec_cfg:
                workers = int(exec_cfg.get("workers", 8))
                workers_src = "config"
            else:
                workers = min(tasks, min(32, cpu))
                workers_src = "default:cpu"
            workers = max(1, min(int(workers), int(tasks)))
            tasks = max(1, int(tasks))

            # --- optional: global pool token sampling (JSONL only) ---
            if global_pool_enabled and eff_token_budget:
                stype0 = source.get("type")
                if stype0 not in ("jsonl", "parquet"):
                    print(
                        f"[warn] {name}: token_budget_mode=global_pool currently supports only jsonl sources; "
                        f"falling back to per-rank limiter",
                        file=sys.stderr,
                    )
                else:
                    pool_dir = out_root / "_pool" / name
                    _stage(
                        f"global_pool sampling enabled: token_budget={int(eff_token_budget):,} tasks={tasks} "
                        f"(writes to {pool_dir})",
                        dataset=name,
                    )
                    if stype0 == "jsonl":
                        _global_pool_sample_jsonl_to_shards(
                            input_files=matched_files,
                            base_adapter=base_adapter,
                            dataset_name=name,
                            out_dir=pool_dir,
                            tasks=tasks,
                            token_budget=int(eff_token_budget),
                            seed=int(eff_seed),
                            chars_per_token=float(eff_chars_per_token),
                            force=bool(force),
                        )
                    else:
                        # parquet -> jsonl pool shards (方案A), built in parallel
                        # `global_pool_workers` controls how many processes build the pool shards (parquet->jsonl).
                        # If not specified, follow -j, else default to CPU count (capped) for better throughput.
                        cpu = int(os.cpu_count() or 1)
                        if "global_pool_workers" in exec_cfg:
                            pool_workers = int(exec_cfg.get("global_pool_workers"))
                        elif "pool_workers" in exec_cfg:
                            pool_workers = int(exec_cfg.get("pool_workers"))
                        elif workers_override is not None:
                            pool_workers = int(workers_override)
                        else:
                            pool_workers = min(32, cpu)
                        pool_workers = max(1, int(pool_workers))
                        _global_pool_sample_parquet_to_shards(
                            input_files=matched_files,
                            adapter_kind=adapter_kind,
                            dataset_name=name,
                            out_dir=pool_dir,
                            tasks=tasks,
                            token_budget=int(eff_token_budget),
                            seed=int(eff_seed),
                            chars_per_token=float(eff_chars_per_token),
                            build_workers=pool_workers,
                            force=bool(force),
                        )
                    # Replace source to read from the pool shards, and disable per-rank budget limiter.
                    source = {"type": "jsonl", "data_dir": pool_dir.as_posix(), "glob": "pool_*.jsonl"}
                    base = Path(source["data_dir"])
                    matched_files = sorted(list(base.glob(source["glob"])))
                    file_count = len(matched_files)
                    try:
                        total_bytes = sum(p.stat().st_size for p in matched_files if p.is_file())
                    except Exception:
                        total_bytes = 0
                    eff_token_budget = None
                    _stage(f"global_pool ready: shard_files={file_count} size≈{(total_bytes/(1024**2)):.1f}MB", dataset=name)
                    # Rebuild reader to point at the pool dir (do not reuse old reader/paths_file).
                    reader = build_reader(source, datasets_root, adapter_fn, paths_file=None)
                    if limit != -1:
                        reader.limit = limit

            # --- pipeline ---
            output_filename = "${rank}.jsonl" if not compression else "${rank}.jsonl.gz"
            pipeline: list = [reader]
            if exclude_ids_dir:
                ids = _get_blacklist_ids()
                if ids:
                    pipeline.append(ExcludeIdsFilter(ids, dataset_name=name, mixed=bool(mixed_name)))
            if eff_token_budget:
                per_rank_budget = int(eff_token_budget)
                if tasks > 1:
                    per_rank_budget = max(1, (int(eff_token_budget) + tasks - 1) // tasks)
                pipeline.append(TokenBudgetLimiter(per_rank_budget, seed=eff_seed, chars_per_token=eff_chars_per_token))

            progress_cfg = exec_cfg.get("progress") or {}
            if not isinstance(progress_cfg, dict):
                progress_cfg = {}
            every_docs = int(progress_cfg.get("every_docs", exec_cfg.get("log_every_docs", 20000)))
            every_seconds = float(progress_cfg.get("every_seconds", exec_cfg.get("log_every_seconds", 30.0)))
            if every_docs > 0 or every_seconds > 0:
                pipeline.append(
                    ProgressLogger(
                        dataset_name=name,
                        every_docs=every_docs,
                        every_seconds=every_seconds,
                        chars_per_token=eff_chars_per_token,
                    )
                )

            if not prepare_only and not mixed_name:
                pipeline.append(
                    StdJsonlWriter(
                        str(ds_out),
                        output_filename=output_filename,
                        compression=compression,
                        max_file_size=writer_max_file_size,
                    )
                )

            if prepare or mixed_name:
                if mixed_name:
                    prepare_out = out_root / "mixed" / mixed_name
                else:
                    prepare_out = out_root / "prepare" / name
                prepare_out.mkdir(parents=True, exist_ok=True)

                def _prepare_adapter(self, document):  # noqa: ANN001
                    uid = str(getattr(document, "id", ""))
                    if mixed_name:
                        uid = f"{name}::{uid}"
                    return {"uuid": uid, "text": getattr(document, "text", "")}

                prepare_filename = f"{name}__${{rank}}.jsonl" if mixed_name else "${rank}.jsonl"
                pipeline.append(
                    StdJsonlWriter(
                        str(prepare_out),
                        output_filename=prepare_filename,
                        compression=None,
                        adapter=_prepare_adapter,
                        max_file_size=writer_max_file_size,
                    )
                )

            required_dirs: list[Path] = []
            if want_prepare:
                if mixed_name:
                    required_dirs.append(out_root / "mixed" / mixed_name)
                else:
                    required_dirs.append(out_root / "prepare" / name)
            if want_full:
                required_dirs.append(ds_out)

            outputs_ok = True
            for d in required_dirs:
                if not _has_any_jsonl(d):
                    outputs_ok = False
                    break

            if force:
                print(f"[force] {name}: deleting {logs_dir} and existing outputs", file=sys.stderr)
                try:
                    shutil.rmtree(logs_dir, ignore_errors=True)
                except Exception:
                    pass
                for d in required_dirs:
                    try:
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception:
                        pass
                logs_dir.mkdir(parents=True, exist_ok=True)
                if prepare:
                    (out_root / "prepare" / name).mkdir(parents=True, exist_ok=True)
                if not prepare_only:
                    ds_out.mkdir(parents=True, exist_ok=True)
            else:
                if logs_dir.exists() and not outputs_ok:
                    print(f"[warn] {name}: outputs missing but logs exist; clearing {logs_dir} to re-run", file=sys.stderr)
                    try:
                        shutil.rmtree(logs_dir, ignore_errors=True)
                    except Exception:
                        pass
                    logs_dir.mkdir(parents=True, exist_ok=True)

            glob_desc = pattern if pattern else (";".join(globs_for_stats) if globs_for_stats else "")
            tb = f"{eff_token_budget}" if eff_token_budget else "0"
            tb_mode = "split" if (eff_token_budget and tasks > 1) else ("global" if eff_token_budget else "off")
            out_mode = (
                f"mixed:{mixed_name}"
                if mixed_name
                else ("prepare-only" if prepare_only else ("prepare+full" if prepare else "full"))
            )
            total_mb_str = f"{(total_bytes / (1024 * 1024)):.1f}MB" if total_bytes else "0MB"
            start_method = str(exec_cfg.get("start_method", exec_cfg.get("mp_start_method", "forkserver")) or "").strip()
            if start_method not in {"forkserver", "fork", "spawn"}:
                print(f"[warn] {name}: unknown executor.start_method={start_method!r}; using 'forkserver'", file=sys.stderr)
                start_method = "forkserver"

            print(
                f"[run] {name} source={source['type']} dir={source['data_dir']} glob={glob_desc} "
                f"files={file_count} size={total_mb_str} tasks={tasks} workers={workers}({workers_src}) "
                f"mp_start={start_method} token_budget={tb}({tb_mode}) out={out_mode}",
                file=sys.stderr,
            )

            # Heartbeat from the main process in case worker processes never start / never print.
            hb_sec = float(exec_cfg.get("heartbeat_seconds", 120.0))
            hb_stop = threading.Event()

            def _hb() -> None:
                if hb_sec <= 0:
                    return
                t0 = time.monotonic()
                while not hb_stop.wait(timeout=hb_sec):
                    dt = time.monotonic() - t0
                    print(f"[hb] {name}: still running (elapsed={dt:.0f}s)", file=sys.stderr)

            threading.Thread(target=_hb, daemon=True).start()

            _stage(
                f"executor starting (LocalPipelineExecutor.run) tasks={tasks} workers={workers} mp_start={start_method} pid={os.getpid()}",
                dataset=name,
            )
            t_exec0 = time.monotonic()
            LocalPipelineExecutor(
                pipeline=pipeline,
                logging_dir=str(logs_dir),
                tasks=tasks,
                workers=workers,
                start_method=start_method,
            ).run()
            hb_stop.set()
            dur_s = time.monotonic() - t_exec0
            if dur_s >= 1.0:
                print(f"[done] {name}: executor finished in {dur_s:.1f}s", file=sys.stderr)
            # datatrove writes one completion marker per rank (when enabled by its executor implementation)
            try:
                comp_dir = Path(logs_dir) / "completions"
                if comp_dir.exists():
                    comp_n = len(list(comp_dir.iterdir()))
                    print(f"[info] {name}: completions={comp_n}/{tasks} ({comp_dir})", file=sys.stderr)
            except Exception:
                pass
            # --- postprocess: merge outputs to a target number of files ---
            merge_to = int(exec_cfg.get("merge_to_files", exec_cfg.get("merge_outputs_to_files", 0)) or 0)
            if merge_to > 0:
                keep_inputs = bool(exec_cfg.get("merge_keep_inputs", False))
                for d in required_dirs:
                    try:
                        # Merge is intentionally single-process and I/O-bound; log it explicitly to avoid confusion
                        # with "executor not parallel".
                        in_files = sorted([p for p in Path(d).glob("*.jsonl") if p.is_file()])
                        in_bytes = 0
                        for p in in_files:
                            try:
                                in_bytes += p.stat().st_size
                            except Exception:
                                pass
                        _stage(
                            f"post-merge starting: dir={d} files={len(in_files)} size≈{(in_bytes/(1024**3)):.2f}GB -> target_files={merge_to} keep_inputs={keep_inputs}",
                            dataset=name,
                        )
                        t_m0 = time.monotonic()
                        merged = _merge_jsonl_folder_to_n(
                            Path(d),
                            target_files=merge_to,
                            keep_inputs=keep_inputs,
                            prefix="merged",
                        )
                        mdur = time.monotonic() - t_m0
                        if merged:
                            print(
                                f"[post] {name}: merged {Path(d)} -> {len(merged)} file(s) (target={merge_to}, keep_inputs={keep_inputs})",
                                file=sys.stderr,
                            )
                        _stage(f"post-merge finished in {mdur:.1f}s", dataset=name)
                    except Exception:
                        print(f"[warn] {name}: post-merge failed for {d}", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
            print(f"[ok] {name} -> {ds_out}", file=sys.stderr)
            return 0
        except Exception:
            print(f"[error] {name}: crashed inside UDatasets backend", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

    dataset_parallelism = max(1, int(dataset_parallelism))
    if dataset_parallelism == 1:
        for cfg in cfgs:
            rc = _run_one(cfg)
            if rc != 0:
                return rc
        return 0

    # Run multiple datasets concurrently to reduce "tail" idle time across datasets.
    # We use threads here (not a process pool) because each dataset run launches its own multiprocessing
    # work via LocalPipelineExecutor; threading avoids pickling constraints and is stable on Linux/WSL.
    # Important: keep total concurrency reasonable to avoid oversubscribing CPU/disk.
    cfgs_run: list[dict] = []
    for cfg in cfgs:
        ds = cfg.get("dataset", {})
        name = (ds.get("name") or "").strip()
        if not name:
            continue
        if only and name != only:
            continue
        cfgs_run.append(cfg)

    if not cfgs_run:
        return 0

    print(f"[info] dataset_parallelism={dataset_parallelism}: running {len(cfgs_run)} dataset(s) concurrently", file=sys.stderr)

    first_err = 0
    with ThreadPoolExecutor(max_workers=dataset_parallelism) as ex:
        fut_to_name = {}
        for cfg in cfgs_run:
            ds = cfg.get("dataset", {})
            name = (ds.get("name") or "").strip() or "<unknown>"
            fut = ex.submit(_run_one, cfg)
            fut_to_name[fut] = name

        for fut in as_completed(fut_to_name):
            name = fut_to_name[fut]
            try:
                rc = int(fut.result())
            except Exception:
                print(f"[error] {name}: crashed (dataset thread)", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                first_err = first_err or 1
                continue
            if rc != 0:
                print(f"[error] {name}: exited with {rc}", file=sys.stderr)
                first_err = first_err or rc

    return int(first_err or 0)