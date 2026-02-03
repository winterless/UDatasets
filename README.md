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
  --workers 20
```

## 输出结构

- **完整 Document 输出**（默认开启，除非 `--mixed`）：`<out-root>/<dataset_name>/*.jsonl`
  - 每行：`{"text": "...", "id": "...", "metadata": {"raw": {...}}}`
- **训练用 prepare 输出**（默认开启；`--mixed` 时写入 mixed 目录）：`<out-root>/prepare/<dataset_name>/*.jsonl`
  - 每行：`{"uuid": "...", "text": "..."}`
- **mixed 输出**（`--mixed NAME`）：`<out-root>/mixed/<NAME>/*.jsonl`
  - 每行：`{"uuid": "<dataset>::<id>", "text": "..."}`（uuid 会加 dataset 前缀防冲突）

## CLI 选项（`python -m cli.runner --help`）

- **`--datasets-root`**：本地数据根目录（默认 `~/workspace/datasets`）
- **`--config-dir`**：配置目录（默认 `configs/datasets`）
- **`--out-root`**：输出根目录（默认 `out/datatrove_documents`）
- **`--workers`**：覆盖并发 worker 进程数（不传则读 `executor.workers`）
- **`--shard-jsonl-mb`**：预分片单个大 JSONL 为多个小 JSONL（写到 `<out-root>/_shards/<dataset>/`），提升并行度并降低单文件被预算截断的风险；`0` 关闭
- **`--system-ratio`**：按比例把 `raw.system`（工具/规范说明）拼进 `text`（按 doc id 稳定抽样）；`0` 关闭
- **`--mixed NAME`**：mixed 模式：所有数据集写到同一个目录 `<out-root>/mixed/NAME/`
- **`--exclude-ids-dir DIR`**：加载黑名单 id/uuid：递归扫描 DIR 下所有文件，收集每行 JSON 的 `uuid/id/doc_id`（或纯文本行）；新输出会跳过这些 id（每次执行都会重新扫描）
- **`--limit N`**：调试用：限制文档条数（`-1` 为全量）

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
  "executor": { "workers": 8, "target_shard_mb": 50 }
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
  - `workers`：并发 worker 进程数（会被限制到 `<= 自动分片数`）
  - 分片数：由“预计输出大小” + `target_shard_mb` 自动决定（若配置了 `token_billions`，会按预算估算输出大小）
- **输出滚动切分**（按文件大小切）：  
  - `target_shard_mb`：目标单文件大小（<=0 表示不切；越小输出文件越多）
- **按 token 预算抽样（配置内控制；CLI 不再提供 `-B`）**：
  - `token_billions`：目标 B（float；内部乘以 1e9）
  - `chars_per_token`：token 估算系数（默认 4.0）

- **后处理（输出文件合并）**：
  - `merge_to_files`（可选，int）：datatrove 跑完后，把输出目录下的 `*.jsonl` / `*.jsonl.gz` **合并**成指定数量的文件（例如 1）。
    - 适用场景：想用更高并行度提速，但最终只想要少量大文件

示例（并行跑，最终合并为 1 个文件）：

```json
{
  "executor": {
    "workers": 192,
    "token_billions": 1,
    "merge_to_files": 1
  }
}
```
