# UData

一个统一的数据归一化入口：把不同来源/格式的数据集转换成标准 `Document`（`text`, `id`, `metadata.raw`），用于后续 CPT / 统一评测流水线接入。

## 目录

- `src/`: 代码（按功能划分目录）
- `configs/datasets/*.json`: 数据集配置（每数据集一个）
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

3) 准备配置（每数据集一个 JSON，放在 `configs/datasets/`，已提供 4 个示例）：
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
  -B 1 \
  --token-budget-parallel \
  -j 20
```

输出示例：
- 完整 Document：`<out-root>/<dataset_name>/00000.jsonl`（每行一个 Document 字典；包含 metadata/raw）
- 训练用 prepare：`<out-root>/prepare/<dataset_name>/00000.jsonl`（每行仅 uuid+text）

默认不压缩输出（`00000.jsonl`）。如需 gzip 压缩，传 `--compression gzip`，则输出为 `00000.jsonl.gz`。

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

### 按 token 预算抽样输出（-B）

如果你想生成固定规模的训练语料（按 `text` 的 token 数量粗略估算），可以用：

- `-B 1`：目标约 **1B tokens**
- `-B 0.2`：目标约 **0.2B tokens**

实现说明：
 - token 数估算是近似：默认大致按 `len(text) / 4` 换算，不追求特别精确（可用 `--chars-per-token` 调整）
- 输出体积（GB）会明显大于“纯 text”的估算，因为 prepare 是 JSONL（包含 uuid/key/引号/转义等开销）
- 达到预算后会提前停止；如果数据不足则输出最大可用数据
- `TokenBudgetLimiter` 是 **按 task(rank) 生效** 的：tasks=1 时最接近“全局预算”；tasks>1 时会把预算按 tasks **均分**，属于“近似全局预算”

如果你希望在 `-B` 预算模式下也并行跑（更快，但预算是“近似全局”），可以加：

```bash
PYTHONPATH=src python -m cli.runner ... -B 1 --token-budget-parallel -j 20
```

并行参数小结：
- `-j/--parallelism N`：同时设置 `--tasks N` 和 `--workers N`（推荐日常使用）
- `--tasks/--workers`：高级用法，分别控制输出分片数量（tasks）和同时并发数（workers）

如需校准体量，可用 `--chars-per-token` 调整粗略换算（例如中文/代码类文本可能更接近 2–3 chars/token）：

```bash
PYTHONPATH=src python -m cli.runner ... -B 1 --chars-per-token 2.5
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

