# UData

一个统一的数据归一化入口：把不同来源/格式的数据集转换成训练友好的 jsonl（`uuid`,`text`），并可选输出完整 datatrove `Document`（`text`,`id`,`metadata.raw`）。

## 目录

- **代码**：`src/`
- **数据集配置**：`configs/datasets/*.json`（每数据集一个）

## 快速开始

```bash
cd /home/unlimitediw/workspace/TYDeepResearch/UDatasets
python -m pip install -e "/home/unlimitediw/workspace/datatrove[io]"

PYTHONPATH=src python -m cli.runner \
  --datasets-root /home/unlimitediw/workspace/datasets \
  --config-dir configs/datasets \
  --out-root out \
  --prepare-only \
  -j 20
```

## 输出结构

- **完整 Document 输出**（默认开启，除非 `--prepare-only` 或 `--mixed`）：`<out-root>/<dataset_name>/*.jsonl[.gz]`
  - 每行：`{"text": "...", "id": "...", "metadata": {"raw": {...}}}`
- **训练用 prepare 输出**（`--prepare` / `--prepare-only` / `--mixed` 会写）：`<out-root>/prepare/<dataset_name>/*.jsonl`
  - 每行：`{"uuid": "...", "text": "..."}`
- **mixed 输出**（`--mixed NAME`）：`<out-root>/mixed/<NAME>/*.jsonl`
  - 每行：`{"uuid": "<dataset>::<id>", "text": "..."}`（uuid 会加 dataset 前缀防冲突）

## CLI 选项（`python -m cli.runner --help`）

- **`--datasets-root`**：本地数据根目录（默认 `~/workspace/datasets`）
- **`--config-dir`**：配置目录（默认 `configs/datasets`）
- **`--out-root`**：输出根目录（默认 `out/datatrove_documents`）
- **`-j/--parallelism`**：便捷并行度，同时设置 `--tasks` 与 `--workers`
- **`-J/--dataset-parallelism`**：多个 dataset config 并发运行（减少“拖尾”）
- **`--tasks`**：覆盖 datatrove tasks（rank）数量（不传则读 `executor.tasks`）
- **`--workers`**：覆盖并发 worker 进程数（不传则读 `executor.workers`）
- **`--shard-jsonl-mb`**：预分片单个大 JSONL 为多个小 JSONL（写到 `<out-root>/_shards/<dataset>/`），提升并行度并降低单文件被预算截断的风险；`0` 关闭
- **`--system-ratio`**：按比例把 `raw.system`（工具/规范说明）拼进 `text`（按 doc id + `--seed` 稳定抽样）；`0` 关闭
- **`--system-max-chars`**：拼入 `raw.system` 的最大字符数上限（避免超长提示重复）
- **`--mixed NAME`**：mixed 模式：所有数据集写到同一个目录 `<out-root>/mixed/NAME/`
- **`--exclude-ids-dir DIR`**：加载黑名单 id/uuid：递归扫描 DIR 下所有文件，收集每行 JSON 的 `uuid/id/doc_id`（或纯文本行）；新输出会跳过这些 id
- **`--prepare`**：额外写一份训练用 prepare 输出（`uuid,text`）
- **`--prepare-only`**：只写 prepare 输出，不写完整 Document（更省 IO/磁盘）
- **`--compression {none,gzip}`**：完整 Document 输出是否 gzip 压缩（prepare/mixed 固定不压缩）
- **`--only <dataset.name>`**：只跑某一个数据集（按 config 的 `dataset.name`）
- **`--limit N`**：调试用：限制文档条数（`-1` 为全量）
- **`--seed`**：用于采样/洗牌（如启用 token budget）和 `--system-ratio` 稳定选择
- **`--force`**：强制重跑：清理该数据集日志状态与输出

## 配置格式（通用）

### 配置开关（enabled）

- **`enabled: true|false`**：只有 `enabled: true` 的配置才会处理（否则跳过）

### 最小配置示例

```json
{
  "enabled": true,
  "dataset": { "name": "my-dataset" },
  "sources": [
    { "type": "jsonl", "data_dir": "my-dataset/data", "glob": "**/*.jsonl" }
  ],
  "adapter": { "kind": "agent-data-collection" },
  "executor": { "tasks": 20, "workers": 8, "min_shard_mb": 2, "target_shard_mb": 50 }
}
```

### sources（输入）字段

- **`type`**：`jsonl` / `parquet` / `csv` / `json_array`
- **`data_dir`**：相对 `--datasets-root` 的目录（也支持绝对路径）
- **`glob`**：匹配输入文件（支持 `**/*.jsonl`）
- **`globs`**（可选）：多个 glob 的 union（会生成内部 `paths.txt`）
- **`path_keywords`**（可选）：仅保留“相对路径包含关键字”的文件（适用于 Math 这种按子目录划分的场景）
  - 也支持别名：`path_keyword`（单个字符串）/ `include_keywords`

### adapter.kind（解析/拼 text 的逻辑）

当前已支持：
- `toucan`
- `agent-data-collection`
- `glaive`
- `hephaestus`
- `nemotron-math-v2`
- `nemotron-pretraining-sft-v1`

### executor（并行/切分/预算）

- **并行**：
  - `tasks`：逻辑分片数（rank 数；通常也决定“至少”输出文件数）
  - `workers`：并发 worker 进程数（会被限制到 `<= tasks`）
- **输出滚动切分**（按文件大小切）：  
  - `min_shard_mb`：输入总大小小于该阈值时不启用滚动切分
  - `target_shard_mb`：启用滚动切分时的目标单文件大小（越小输出文件越多）
- **按 token 预算抽样（配置内控制；CLI 不再提供 `-B`）**：
  - `token_budget`：目标 token 数（int）
  - 或 `token_billions`：目标 B（float；内部乘以 1e9）
  - `token_budget_parallel`：true/false（true 时允许并行 tasks，预算会按 task 拆分，近似全局）
  - `chars_per_token`：token 估算系数（默认 4.0）
  - `seed`：采样/洗牌 seed（默认继承 CLI `--seed`）
  - `token_budget_mode`（可选）：
    - `per_rank`（默认行为）：使用 `TokenBudgetLimiter`，预算是 **按 rank** 生效；tasks>1 时会把预算拆分到每个 rank
    - `global_pool`（jsonl/parquet）：先把该数据集所有输入文件加入一个“全局池”，按 seed 稳定洗牌后流式抽样到目标 token，**预写**为 `tasks` 个临时 shard：
      - `<out-root>/_pool/<dataset_name>/pool_*.jsonl`
      - 然后 datatrove 再读取这些 shard 并输出，因此最终通常会得到 **m=tasks 个输出文件**
      - 这个模式用于避免“单个大文件被分配到某个 rank 导致被 token 预算截断”的问题
    - 也支持别名：`global_token_pool: true`（等价于 `token_budget_mode: "global_pool"`）
  - `global_pool_workers`（可选，仅 parquet 时有明显意义）：构建 `_pool` 临时 shard 时的并行进程数（默认跟随 CLI `-j/--parallelism` 里的 workers）

- **后处理（输出文件合并）**：
  - `merge_to_files`（可选，int）：datatrove 跑完后，把输出目录下的 `*.jsonl` / `*.jsonl.gz` **合并**成指定数量的文件（例如 1）。
    - 适用场景：想用高 `tasks` 提速，但最终只想要少量大文件
  - `merge_keep_inputs`（可选，bool，默认 false）：合并后是否保留原先的分片文件

示例（并行跑，最终合并为 1 个文件）：

```json
{
  "executor": {
    "tasks": 192,
    "workers": 192,
    "token_billions": 1,
    "token_budget_mode": "global_pool",
    "merge_to_files": 1
  }
}
```
