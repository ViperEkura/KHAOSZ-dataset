# DataPipeline

用于训练 KHAOSZ 模型的数据集处理工具。提供文本导出、Tokenize、序列打包、H5 存储等独立工具，支持预训练 / SFT / DPO 三种训练范式。

## 项目结构

```
pipeline/
├── tokenizer.py      # BPE 分词器
├── text.py           # 文本规范化
├── packing.py        # 序列打包（bin-packing）
├── io.py             # 文件/HDF5 读写
├── processors.py     # PT / SFT / DPO 处理器
├── export.py         # Dataset → JSONL 导出
└── cache.py          # JSONL → Tokenize → H5 缓存

pre_train/                          # 预训练数据处理脚本
supervised_finetuning/              # SFT 数据处理脚本
reforce_learning/                   # DPO 数据处理脚本
```

## 设计理念

每个模块**独立可用、零互相依赖**，调用者按需组合：

```
HuggingFace Hub
      │
      ▼  load_dataset()
  DatasetDict
      │
      ▼  export_dataset()          ← pipeline/export.py
  JSONL 文件
      │
      ▼  cache_jsonl()            ← pipeline/cache.py
      │     ├─ Processor.process()    ← pipeline/processors.py
      │     ├─ SequencePacker.pack()  ← pipeline/packing.py
      │     └─ IOHandler.save_h5()    ← pipeline/io.py
  HDF5 张量文件
```

> 各阶段之间通过磁盘文件解耦。你可以只执行阶段 1（导出 JSONL），也可以继续执行阶段 2（tokenize 并缓存为 H5），按需选择。

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 阶段 1：导出数据集为 JSONL

```python
from datasets import load_dataset
from pipeline.export import export_dataset

dataset = load_dataset("your-dataset")
export_dataset(
    dataset=dataset["train"],        # 直接传 Dataset，不传 DatasetDict
    output_dir="./dataset",
    output_prefix="my-data",
    max_chunks=5,                    # 可选，限制 chunk 数量（调试用）
)
```

**自定义转换函数：**

```python
def process_func(example):
    # 提取字段、转换格式、展开多轮对话等
    return {"query": example["instruction"], "response": example["output"]}

export_dataset(
    dataset=dataset["train"],
    output_dir="./dataset",
    output_prefix="my-sft",
    process_func=process_func,
)
```

> `process_func` 返回单个 `dict` 或 `list[dict]`（一条样本可展开为多条）。

**使用文本规范化：**

```python
from pipeline.text import TextNormalizer

normalizer = TextNormalizer()

def process_func(example):
    return {"text": normalizer.normalize(example["content"])}

export_dataset(
    dataset=dataset["train"],
    output_dir="./dataset",
    output_prefix="my-pretrain",
    process_func=process_func,
)
```

### 阶段 2：Tokenize 并缓存为 HDF5

```python
from pipeline.tokenizer import BpeTokenizer
from pipeline.processors import ProcessorFactory
from pipeline.cache import cache_jsonl

tokenizer = BpeTokenizer("tokenizer.json")
processor = ProcessorFactory.create("pt", tokenizer)

cache_jsonl(
    files=["./dataset/my-pretrain_chunk_0.jsonl"],
    output_dir="./cached",
    processor=processor,
    pack_size=4096,       # 可选，打包长度；<=0 不打包
    pad_value=1,
)
```

**处理器类型：**

| 类型 | 工厂 key | 输入格式 | 输出 keys |
|------|----------|---------|-----------|
| 预训练 | `"pt"` | `{"text": "..."}` | `["sequence"]` |
| SFT | `"sft"` | `{"query": "...", "response": "..."}` | `["sequence", "loss_mask"]` |
| DPO | `"dpo"` | 待定 | `["chosen", "chosen_mask", "rejected", "rejected_mask"]` |

## 独立工具参考

### BpeTokenizer

```python
from pipeline.tokenizer import BpeTokenizer

tokenizer = BpeTokenizer("tokenizer.json")
ids = tokenizer.encode("hello world")           # → [1234, 5678, 1]
text = tokenizer.decode(ids)                     # → "hello world"
len(tokenizer)                                   # → 词表大小
```

### TextNormalizer

```python
from pipeline.text import TextNormalizer

normalizer = TextNormalizer()
text = normalizer.normalize(raw_text)
```

替换规则包括：全角引号 → 半角、各种短横线统一、不间断空格 → 普通空格等。支持自定义规则：

```python
normalizer = TextNormalizer(custom_rules={"旧词": "新词"})
```

### SequencePacker

```python
from pipeline.packing import SequencePacker

packer = SequencePacker(pack_size=4096, pad_value=0)
packed = packer.pack(list_of_tensors)    # → List[Tensor]，每个长度为 pack_size
```

### IOHandler

```python
from pipeline.io import IOHandler

# 保存
IOHandler.save_h5("./output", "my_data", {"sequence": [tensor1, tensor2]})

# 加载
data = IOHandler.load_h5("./output")     # → {"sequence": [tensor1, tensor2, ...]}

# 遍历文件
files = IOHandler.fetch_files("./dataset")
folders = IOHandler.fetch_folders("./dataset")
```

### 自定义 Processor

```python
from pipeline.processors import BaseProcessor, ProcessorFactory
import torch

class MyProcessor(BaseProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, input_dict):
        tokens = self.tokenizer.encode(input_dict["text"])
        return {"sequence": torch.tensor(tokens, dtype=torch.int32)}

    @property
    def output_keys(self):
        return ["sequence"]

ProcessorFactory.register("my_type", MyProcessor)
```

## 运行脚本

```bash
# 预训练
python pre_train/chinese-c4.py
python pre_train/english-wiki.py

# SFT
python supervised_finetuning/sft_belle.py
python supervised_finetuning/sft_coder.py

# DPO
python reforce_learning/dpp_chinese_dpo_pairs.py
```

## 输出格式

**JSONL**（阶段 1 输出）：

```jsonl
{"text": "训练文本内容..."}
{"query": "问题", "response": "答案"}
```

**HDF5**（阶段 2 输出）：

```
my_data.h5
├── sequence/
│   ├── data_0    # Tensor (4096,) int32
│   ├── data_1    # Tensor (4096,) int32
│   └── ...
└── loss_mask/     # 仅 SFT
    ├── data_0    # Tensor (4096,) bool
    └── ...
```
