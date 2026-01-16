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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _run_one(cfg: dict) -> int:
        ds = cfg.get("dataset", {})
        name = (ds.get("name") or "").strip()
        if not name:
            return 0
        if only and name != only:
            return 0

        exec_cfg = cfg.get("executor") or {}

        # Token-budget sampling is controlled via config (executor.*), with optional global overrides
        # through the backend API (token_budget/token_budget_parallel/seed/chars_per_token).
        #
        # Supported keys:
        # - executor.token_budget: int tokens
        # - executor.token_billions: float B (converted to tokens via *1e9)
        # - executor.token_budget_parallel: bool (split budget across tasks for more parallelism; approximate global budget)
        # - executor.seed: int
        # - executor.chars_per_token: float (tokens ~= len(text)/chars_per_token)
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

        adapter_kind = (cfg.get("adapter", {}) or {}).get("kind", "")
        base_adapter = BASE_ADAPTERS.get(adapter_kind)
        if not base_adapter:
            print(f"[skip] {name}: unknown adapter kind {adapter_kind!r} ({cfg.get('_config_path')})", file=sys.stderr)
            return 0
        adapter_fn = make_adapter(
            base_adapter,
            name,
            cfg.get("_config_path", ""),
            system_ratio=float(system_ratio),
            system_max_chars=int(system_max_chars),
            seed=int(eff_seed),
        )

        source = pick_source(cfg, datasets_root)
        if not source:
            print(f"[skip] {name}: no usable source found (paths missing or deps missing)", file=sys.stderr)
            return 0
        if eff_token_budget:
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
                return 0
            paths_file_path = (out_root / "_paths" / name / "paths.txt")
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

        def _estimate_jsonl_file_tokens(p: Path, *, sample_lines: int = 200) -> float | None:
            """
            Estimate tokens for a JSONL file by sampling a few lines and extrapolating.
            This is used as a guardrail to avoid per-task token budget truncating a single huge file when tasks>1.
            Returns None if the file can't be sampled.
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
            # - If token budget sampling is enabled, default is tasks=1 for the closest-to-true global budget.
            # - If token_budget_parallel is enabled, we allow parallel tasks (budget will be split across tasks).
            if eff_token_budget and not eff_token_budget_parallel:
                tasks = 1
            else:
                # If input is small, don't bother splitting into multiple ranks/tasks.
                if total_bytes < min_bytes or file_count <= 1:
                    tasks = 1
                else:
                    # Default parallelism: cap to number of files to avoid empty ranks.
                    default_tasks = int(exec_cfg.get("default_tasks", 20))
                    tasks = max(1, min(file_count, default_tasks))

        # Guardrail: for disk readers, tasks beyond number of input files just creates idle ranks.
        # This is especially common on smaller machines where users set -j very high.
        if file_count > 0 and tasks > file_count:
            print(f"[warn] {name}: tasks={tasks} > input_files={file_count}; capping tasks to {file_count}", file=sys.stderr)
            tasks = file_count

        # Guardrail: when running with a per-task token budget (token_budget + tasks>1),
        # a single huge input file can be assigned to a single rank, causing that file to be truncated.
        # For JSONL sources, estimate the largest file and cap tasks so per-rank budget can cover it.
        if eff_token_budget and tasks > 1 and source.get("type") == "jsonl":
            # per-rank budget is split when tasks>1
            per_rank_budget = max(1, (int(eff_token_budget) + tasks - 1) // tasks)
            est_max_file_tokens = 0.0
            # sample only a handful of largest files for speed
            for p in sorted(matched_files, key=lambda x: getattr(x.stat(), "st_size", 0), reverse=True)[:5]:
                est = _estimate_jsonl_file_tokens(p)
                if est is not None:
                    est_max_file_tokens = max(est_max_file_tokens, float(est))
            if est_max_file_tokens > 0 and per_rank_budget < est_max_file_tokens:
                # compute max tasks allowed to avoid truncating the largest file
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

        workers = int(workers_override) if workers_override is not None else int(exec_cfg.get("workers", 8))
        workers = max(1, min(workers, tasks))
        tasks = max(1, tasks)

        output_filename = "${rank}.jsonl" if not compression else "${rank}.jsonl.gz"
        pipeline = [reader]
        if eff_token_budget:
            # NOTE: TokenBudgetLimiter is per-rank. We treat `token_budget` as a desired *global* budget.
            # - tasks=1: closest to a true global budget
            # - tasks>1: approximate by splitting the budget across tasks (may still slightly overshoot)
            per_rank_budget = int(eff_token_budget)
            if tasks > 1:
                per_rank_budget = max(1, (int(eff_token_budget) + tasks - 1) // tasks)
            pipeline.append(TokenBudgetLimiter(per_rank_budget, seed=eff_seed, chars_per_token=eff_chars_per_token))

        # Main output: full Document dict (includes metadata/raw)
        if not prepare_only and not mixed_name:
            pipeline.append(
                StdJsonlWriter(
                    str(ds_out),
                    output_filename=output_filename,
                    compression=compression,
                    max_file_size=writer_max_file_size,
                )
            )

        # Prepare output: CPT-friendly lightweight view (uuid + text only)
        if prepare or mixed_name:
            # In mixed mode, write all datasets into a single folder under <out-root>/mixed/<mixed_name>/.
            # We keep prepare schema {uuid,text} and prefix uuid with '<dataset>::' to avoid collisions.
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

        def _has_any_jsonl(folder: Path) -> bool:
            if not folder.exists():
                return False
            # includes rolling shards like 000_00000.jsonl and plain 00000.jsonl
            return any(folder.glob("*.jsonl")) or any(folder.glob("*.jsonl.gz")) or any(folder.glob("*.jsonl.*"))

        # Force / resume guard:
        # Datatrove will skip tasks if it thinks they are completed based on logs_dir state.
        # If outputs were deleted (or output mode changed, e.g. switching to --prepare-only),
        # we must also clear logs_dir to force a re-run.
        required_dirs: list[Path] = []
        if mixed_name:
            required_dirs.append(out_root / "mixed" / mixed_name)
        elif prepare:
            required_dirs.append(out_root / "prepare" / name)
        if not prepare_only:
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
            # If logs exist but outputs are missing, clear logs to avoid "already completed" no-op runs.
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
        out_mode = (f"mixed:{mixed_name}" if mixed_name else ("prepare-only" if prepare_only else ("prepare+full" if prepare else "full")))
        total_mb_str = f"{(total_bytes / (1024 * 1024)):.1f}MB" if total_bytes else "0MB"
        print(
            f"[run] {name} source={source['type']} dir={source['data_dir']} glob={glob_desc} "
            f"files={file_count} size={total_mb_str} tasks={tasks} workers={workers} token_budget={tb}({tb_mode}) out={out_mode}",
            file=sys.stderr,
        )
        LocalPipelineExecutor(pipeline=pipeline, logging_dir=str(logs_dir), tasks=tasks, workers=workers).run()
        print(f"[ok] {name} -> {ds_out}", file=sys.stderr)
        return 0

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
            except Exception as e:
                print(f"[error] {name}: crashed ({e})", file=sys.stderr)
                first_err = first_err or 1
                continue
            if rc != 0:
                print(f"[error] {name}: exited with {rc}", file=sys.stderr)
                first_err = first_err or rc

    return int(first_err or 0)