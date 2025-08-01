## KHAOSZ-dataset

用于训练的KHAOSZ数据集
### 项目结构

``` bash
.
├── pre_train/                # 预训练数据处理脚本
│   ├── chinese-c4.py         # 中文预训练数据处理
│   ├── chinese-cosmopedia.py # 中文维基数据处理
│   ├── english-fineweb.py    # 英文网页数据处理
│   └── english-wiki.py       # 英文维基数据处理
├── supervised_finetuning/    # 监督微调数据处理脚本
│   ├── sft_belle.py          # BelleGroup对话数据处理
│   ├── sft_chinese_instruct.py # 中文指令微调数据处理
│   ├── sft_coder.py          # 编程对话数据处理
│   └── sft_magpie-pro-300k.py # Magpie对话数据处理
├── modules/
│   ├── utils.py              # 核心数据处理工具（规范化/打包/序列化）
│   └── tokenizer.py          # BPE分词器实现（支持自定义特殊token）
├── dump_file/
│   ├── dump_pt_file.py       # 预训练数据二进制打包
│   └── dump_sft_file.py      # 微调数据二进制打包
├── run.py                    # 主执行脚本（支持subprocess调用）
└── README.md                 # 项目文档
```


### 数据集特性
- 支持多语言混合训练（中/英文）
- 包含以下预训练数据源：
  - Chinese-C4
  - Chinese-Cosmopedia
  - English-Fineweb
  - English-Wiki
- 支持监督微调数据集：
  - Ling-Coder-SFT
  - Chinese-Instruct
  - BelleGroup
  - Magpie-Pro-300K

### 使用说明
1. 安装依赖：
```bash
pip install datasets tokenizers tqdm torch
```
运行数据处理：

```bash
python run.py
```