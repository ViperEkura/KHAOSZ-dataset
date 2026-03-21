"""将 HuggingFace Dataset 分块导出为 JSONL 文件"""
import json
import os
from typing import Callable, Optional, List, Union


def export_dataset(
    dataset,
    output_dir: str,
    output_prefix: str,
    *,
    chunk_size: int = 1_000_000,
    max_chunks: Optional[int] = None,
    process_func: Optional[Callable] = None,
    column: str = "text",
) -> List[str]:
    """
    将 HuggingFace Dataset 分块导出为 JSONL 文件。

    Args:
        dataset: HuggingFace Dataset 对象
        output_dir: 输出目录
        output_prefix: 输出文件名前缀，如 "chinese-c4-pretrain"
        chunk_size: 每个文件的最大样本数
        max_chunks: 最多处理几个 chunk（用于调试）
        process_func: 单条样本的转换函数 (dict) -> dict | list[dict]
        column: 默认提取的文本列名（仅在 process_func 为 None 时使用）

    Returns:
        生成的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(dataset)
    num_chunks = (total + chunk_size - 1) // chunk_size
    lim = min(max_chunks, num_chunks) if max_chunks else num_chunks

    output_files: List[str] = []
    for i in range(lim):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = dataset.select(range(start, end))

        path = os.path.join(output_dir, f"{output_prefix}_chunk_{i}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for example in chunk:
                processed = process_func(example) if process_func else {column: example[column]}
                items = processed if isinstance(processed, list) else [processed]
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        output_files.append(path)
        print(f"[{i + 1}/{lim}] Saved {path}")

    return output_files
