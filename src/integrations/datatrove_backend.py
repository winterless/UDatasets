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
import sys
from pathlib import Path
import random
from typing import Callable, Dict, IO, Literal


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
                break
            consumed += t
            yielded_any = True
            yield doc
            if consumed >= self.token_budget:
                break


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

def make_adapter(base_adapter: Callable, dataset_name: str, config_path: str):
    def _adapter(self, data, path, id_in_file):
        doc = base_adapter(self, data, path, id_in_file)
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

    for cfg in cfgs:
        ds = cfg.get("dataset", {})
        name = (ds.get("name") or "").strip()
        if not name:
            continue
        if only and name != only:
            continue

        adapter_kind = (cfg.get("adapter", {}) or {}).get("kind", "")
        base_adapter = BASE_ADAPTERS.get(adapter_kind)
        if not base_adapter:
            print(f"[skip] {name}: unknown adapter kind {adapter_kind!r} ({cfg.get('_config_path')})", file=sys.stderr)
            continue
        adapter_fn = make_adapter(base_adapter, name, cfg.get("_config_path", ""))

        source = pick_source(cfg, datasets_root)
        if not source:
            print(f"[skip] {name}: no usable source found (paths missing or deps missing)", file=sys.stderr)
            continue
        if token_budget:
            # enable file-level shuffling to make early-stop sampling less biased
            source = dict(source)
            source["_shuffle_files"] = True

        # Support multi-glob sources via a generated paths file (union of patterns).
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
                        rel_paths.append(p.relative_to(base).as_posix())
            rel_paths = sorted(set(rel_paths))
            if not rel_paths:
                print(f"[skip] {name}: globs matched 0 files under {base}", file=sys.stderr)
                continue
            paths_file_path = (out_root / "_logs" / name / "paths.txt")
            paths_file_path.parent.mkdir(parents=True, exist_ok=True)
            paths_file_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
            paths_file = str(paths_file_path)

        reader = build_reader(source, datasets_root, adapter_fn, paths_file=paths_file)
        if limit != -1:
            reader.limit = limit

        ds_out = out_root / name
        logs_dir = out_root / "_logs" / name
        ds_out.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        exec_cfg = cfg.get("executor") or {}
        base = datasets_root / source["data_dir"]
        pattern = (source.get("glob") or "").strip()
        globs_for_stats = source.get("globs") or []
        if not isinstance(globs_for_stats, list):
            globs_for_stats = []
        globs_for_stats = [str(g).strip() for g in globs_for_stats if str(g).strip()]

        if globs_for_stats:
            matched_files = []
            for g in globs_for_stats:
                matched_files.extend([p for p in base.glob(g) if p.is_file()])
        else:
            matched_files = [p for p in base.glob(pattern) if p.is_file()] if pattern else []
        file_count = len(matched_files)
        total_bytes = 0
        for p in matched_files:
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass

        # Output splitting: aim for ~10–15MB per output file by rolling writer files.
        # NOTE: this is the only way to get ~1000 output files from a dataset with few large input files
        # (tasks can never exceed the number of input files for disk readers).
        # Default rolling shard sizing:
        # - enable splitting when total input is >= ~2MB
        # - target output file size around ~2MB
        # (can be overridden per-dataset via config.executor.{min_shard_mb,target_shard_mb})
        min_mb = float(exec_cfg.get("min_shard_mb", exec_cfg.get("min_mb", 2)))
        target_mb = float(exec_cfg.get("target_shard_mb", exec_cfg.get("target_mb", 2)))
        min_bytes = int(min_mb * 1024 * 1024)
        target_bytes = max(int(target_mb * 1024 * 1024), 1)
        writer_max_file_size = -1 if total_bytes < min_bytes else target_bytes

        if tasks_override is not None:
            tasks = int(tasks_override)
        elif "tasks" in exec_cfg:
            tasks = int(exec_cfg.get("tasks", 20))
        else:
            # Default parallelism:
            # - If token_budget is set, default is tasks=1 for the closest-to-true global budget.
            # - If token_budget_parallel is enabled, we allow parallel tasks (budget will be split across tasks).
            if token_budget and not token_budget_parallel:
                tasks = 1
            else:
                # If input is small, don't bother splitting into multiple ranks/tasks.
                if total_bytes < min_bytes or file_count <= 1:
                    tasks = 1
                else:
                    # Default parallelism: cap to number of files to avoid empty ranks.
                    default_tasks = int(exec_cfg.get("default_tasks", 20))
                    tasks = max(1, min(file_count, default_tasks))

        workers = int(workers_override) if workers_override is not None else int(exec_cfg.get("workers", 8))
        workers = max(1, min(workers, tasks))
        tasks = max(1, tasks)

        output_filename = "${rank}.jsonl" if not compression else "${rank}.jsonl.gz"
        pipeline = [reader]
        if token_budget:
            # NOTE: TokenBudgetLimiter is per-rank. We treat `token_budget` as a desired *global* budget.
            # - tasks=1: closest to a true global budget
            # - tasks>1: approximate by splitting the budget across tasks (may still slightly overshoot)
            per_rank_budget = int(token_budget)
            if tasks > 1:
                per_rank_budget = max(1, (int(token_budget) + tasks - 1) // tasks)
            pipeline.append(TokenBudgetLimiter(per_rank_budget, seed=seed, chars_per_token=chars_per_token))

        # Main output: full Document dict (includes metadata/raw)
        if not prepare_only:
            pipeline.append(
                StdJsonlWriter(
                    str(ds_out),
                    output_filename=output_filename,
                    compression=compression,
                    max_file_size=writer_max_file_size,
                )
            )

        # Prepare output: CPT-friendly lightweight view (uuid + text only)
        if prepare:
            prepare_out = out_root / "prepare" / name
            prepare_out.mkdir(parents=True, exist_ok=True)

            def _prepare_adapter(self, document):  # noqa: ANN001
                return {"uuid": document.id, "text": document.text}

            pipeline.append(
                StdJsonlWriter(
                    str(prepare_out),
                    output_filename="${rank}.jsonl",
                    compression=None,
                    adapter=_prepare_adapter,
                    max_file_size=writer_max_file_size,
                )
            )

        glob_desc = pattern if pattern else (";".join(globs_for_stats) if globs_for_stats else "")
        tb = f"{token_budget}" if token_budget else "0"
        tb_mode = "split" if (token_budget and tasks > 1) else ("global" if token_budget else "off")
        out_mode = "prepare-only" if prepare_only else ("prepare+full" if prepare else "full")
        print(
            f"[run] {name} source={source['type']} dir={source['data_dir']} glob={glob_desc} "
            f"tasks={tasks} workers={workers} token_budget={tb}({tb_mode}) out={out_mode}",
            file=sys.stderr,
        )
        LocalPipelineExecutor(pipeline=pipeline, logging_dir=str(logs_dir), tasks=tasks, workers=workers).run()
        print(f"[ok] {name} -> {ds_out}", file=sys.stderr)

    return 0

