# DataPipeline

用于训练KHAOSZ模型的数据集处理工具

## 项目结构

```
khaosz_dataset/
├── modules/                    # 核心模块
│   ├── tokenizer.py           # BPE Tokenizer
│   └── datapipeline/          # 数据管道模块
│       ├── pipeline.py        # 主数据管道
│       ├── processors.py      # 数据处理器（策略模式）
│       ├── io.py             # 文件IO操作
│       ├── packing.py        # 序列打包
│       └── text.py           # 文本规范化
├── tokenizer.json             # Tokenizer配置
└── pyproject.toml            # 项目依赖
```

## 架构设计

本项目采用模块化设计，应用了多种设计模式：

### 设计模式
1. **策略模式** - 不同类型的数据处理器（PT/SFT/DPO）
2. **工厂模式** - 处理器工厂统一创建实例
3. **模板方法模式** - 数据管道流程标准化

### 核心组件

#### DataPipeline（数据管道）
主数据管道，负责数据集的分块处理、格式转换和存储。

```python
from modules.datapipeline import DataPipeline

pipeline = DataPipeline(output_dir="./dataset")
pipeline.process_dataset(
    dataset_dict=dataset,
    output_subdir="my-data"
)
```

#### ProcessorFactory（处理器工厂）
创建不同类型的数据处理器。

```python
from modules.datapipeline import ProcessorFactory
from modules.tokenizer import BpeTokenizer

tokenizer = BpeTokenizer("tokenizer.json")

# 创建预训练处理器
processor = ProcessorFactory.create("pt", tokenizer)

# 创建SFT处理器
processor = ProcessorFactory.create("sft", tokenizer)
```

#### TextNormalizer（文本规范化）
文本预处理和规范化。

```python
from modules.datapipeline import TextNormalizer

normalizer = TextNormalizer()
normalized_text = normalizer.normalize(text)
```

## 使用说明

### 安装依赖

```bash
pip install datasets tokenizers tqdm torch h5py
```

### 运行数据处理

运行所有数据处理脚本：

```bash
python run.py
```

运行特定脚本：

```bash
# 预训练数据处理
python pre_train/english-wiki.py

# SFT数据处理
python supervised_finetuning/sft_belle.py

# DPO数据处理
python reforce_learning/dpp_chinese_dpo_pairs.py
```

### 自定义数据处理

#### 1. 基础数据处理

```python
from datasets import load_dataset
from modules.datapipeline import DataPipeline

# 加载数据集
dataset = load_dataset("your-dataset")

# 创建管道
pipeline = DataPipeline()

# 处理数据
pipeline.process_dataset(
    dataset_dict=dataset,
    output_subdir="output-dir",
    process_func=lambda x: {"text": x["content"]}
)
```

#### 2. 自定义处理器

```python
from modules.datapipeline.processors import BaseProcessor
from modules.datapipeline import ProcessorFactory

class MyProcessor(BaseProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input_dict: dict) -> dict:
        # 自定义处理逻辑
        return {"processed": data}
    
    @property
    def output_keys(self) -> list:
        return ["processed"]

# 注册处理器
ProcessorFactory.register("my_type", MyProcessor)
```

#### 3. 文本规范化

```python
from modules.datapipeline import TextNormalizer

# 使用默认规则
normalizer = TextNormalizer()
text = normalizer.normalize(text)

# 自定义规则
custom_rules = {"旧词": "新词"}
normalizer = TextNormalizer(custom_rules)
```

## 数据输出格式

### JSONL格式
每个数据块保存为JSONL文件：
```
{"text": "训练文本内容..."}
{"query": "问题", "response": "答案"}
```

### H5格式
打包后的张量数据保存为HDF5格式，支持高效加载：
```python
from modules.datapipeline import IOHandler

# 加载H5数据
data = IOHandler.load_h5("./cached_data")
```