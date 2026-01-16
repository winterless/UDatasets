# UData

一个统一的数据归一化入口：把不同来源/格式的数据集转换成标准 `Document`（`text`, `id`, `metadata.raw`），用于后续 CPT / 统一评测流水线接入。

## 目录

- `src/`: 代码（按功能划分目录）
- `configs/datasets/*.json`: 数据集配置（每数据集一个）

### 配置开关（enabled）
每个 `configs/datasets/*.json` 都支持：

- `enabled: true|false`（默认 true）

只有 `enabled: true` 的配置才会被处理；设为 false 会被跳过。
  - 本工程只通过 datatrove backend 的“输入/输出”与 datatrove 交互（见 `src/integrations/datatrove_backend.py`）

## 统一入口：Datatrove 驱动（输出标准 Document：text/id/metadata.raw）

目标：对多套数据（Toucan/agent-data-collection/glaive/Hephaestus）生成 datatrove `Document`：
- `text`: 可训练文本视图
- `id`: 优先使用样本自带 id/uuid，否则用 `path/row_index`
- `metadata.raw`: **原始 record 全量保留**（非 text 字段都在这里）

## 使用方法

1) 进入仓库根目录：

```bash
cd /home/unlimitediw/workspace/TYDeepResearch/UDatasets
```

2) 安装本地 datatrove（不改源码，只安装依赖）：

```bash
python -m pip install -e "/home/unlimitediw/workspace/datatrove[io]"
```

3) 准备配置（每数据集一个 JSON，放在 `configs/datasets/`）：
- `configs/datasets/toucan_1_5m.json`（parquet/jsonl 二选一，优先 parquet）
- `configs/datasets/agent_data_collection.json`
- `configs/datasets/glaive_function_calling_v2.json`
- `configs/datasets/hephaestus_forge.json`

4) 运行（datasets_root 指向你本地数据目录）：

```bash
PYTHONPATH=src python -m cli.runner \
  --datasets-root /home/unlimitediw/workspace/datasets \
  --config-dir configs/datasets \
  --out-root out \
  --prepare-only \
  --system-ratio 0.5 \
  -j 20 \
  -J 2 \
  --mixed mymix
```

输出示例：
- 完整 Document：`<out-root>/<dataset_name>/00000.jsonl`（每行一个 Document 字典；包含 metadata/raw）
- 训练用 prepare：`<out-root>/prepare/<dataset_name>/00000.jsonl`（每行仅 uuid+text）

默认不压缩输出（`00000.jsonl`）。如需 gzip 压缩，传 `--compression gzip`，则输出为 `00000.jsonl.gz`。


### 去重/跳过已存在的 id/uuid（黑名单文件夹）
如果你有一批历史产物（比如 `out/prepare/<dataset>/` 或 `out/mixed/<name>/`），想在新一轮生成时跳过这些样本，可以传：

- `--exclude-ids-dir <DIR>`：递归扫描 `<DIR>` 下所有文件，收集每行 JSON 的 `uuid/id/doc_id`（或纯文本行）作为黑名单，新的输出将跳过这些 id。

提示：在 `--mixed` 模式下，历史 uuid 往往是 `<dataset>::<id>`，黑名单会同时匹配 raw id 和这种带前缀的 uuid。
### 训练用 prepare 输出（仅 uuid + text）

如果你要做 CPT（继续预训练），通常只需要 `text`（以及可选的样本 id）。
可以加 `--prepare`，会在 `<out-root>/prepare/<dataset>/` 下生成对应的 jsonl 分片文件，
每行仅包含：

- `uuid`: 等价于 datatrove 的 `id`
- `text`: 训练文本

示例：`out/prepare/Toucan-1.5M/00000.jsonl`

如果你只需要训练用的 prepare 输出、并且不想额外写一份“完整 Document”（更慢、占更多磁盘），可以用：

```bash
PYTHONPATH=src python -m cli.runner ... --prepare-only
```

### 按 token 预算抽样输出（配置内控制）
如果你想限制单个数据集输出规模，请在对应 config 的 `executor` 里写：

- `token_budget`: 目标 token 数（int）
- 或 `token_billions`: 目标 B（float，内部会乘以 1e9）
- `token_budget_parallel`: true/false（true 时允许并行 tasks，预算会按 task 拆分，近似全局）
- `chars_per_token`: token 估算系数（默认 4.0）
- `seed`: 采样/洗牌的 seed

示例（在某个 `configs/datasets/*.json` 内）：

```json
{
  "executor": {
    "token_billions": 0.2,
    "token_budget_parallel": true,
    "chars_per_token": 4.0,
    "seed": 42
  }
}
```

### 并行参数（tasks/workers）的默认策略

推荐日常用 `-j/--parallelism N`（同时设置 tasks/workers）。

如果命令行未指定 `--tasks/--workers`，会尝试从每个数据集配置里读取：

```json
{
  "executor": { "tasks": 20, "workers": 8 }
}
```

优先级：**CLI 显式传参（`-j`/`--tasks`/`--workers`）> config.executor > 兜底默认值（tasks=20, workers=8）**。

补充：如果 **CLI 未传 `--tasks` 且 config 里也没写 `executor.tasks`**：
- 并行 tasks 会取一个保守默认值（不超过输入文件数，避免大量空 `000xx.jsonl`）
- 想要输出按大小切分（例如每个文件约 **10–15MB**），不是靠 tasks，而是靠 writer 的滚动分片：当输入总大小 ≥ ~10MB 时，会启用 `max_file_size`，输出会变成 `000_00000.jsonl`, `001_00000.jsonl`… 这种多文件形式

### 输入约定（不是统一 jsonl）

“输入格式”由每个数据集对应的 JSON 配置决定（reader）：
- Toucan-1.5M：parquet 或 jsonl（等价；有 `pyarrow` 时优先 parquet）
- agent-data-collection：jsonl
- glaive-function-calling-v2：json（顶层数组 `[...]`）
- Hephaestus-Forge：json（顶层数组 `[...]`）
