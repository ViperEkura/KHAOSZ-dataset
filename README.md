## KHAOSZ-dataset

用于训练的KHAOSZ数据集
### 项目结构

``` bash
.
│   .gitignore
│   dump_pt_file.py
│   dump_sft_file.py
│   README.md
│   run.py
│   test.py
│   tokenizer.json
│
├───modules
│   │   tokenizer.py
│   └───utils.py
│
├───pre_train
│       chinese-c4.py
│       chinese-cosmopedia.py
│       english-fineweb.py
│       english-wiki.py
│
├───reforce_learning
│       dpp_chinese_dpo_pairs.py
│
└───supervised_finetuning
        sft_belle.py
        sft_chinese_instruct.py
        sft_coder.py
        sft_magpie-pro-300k.py
        sft_small_talk.py
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