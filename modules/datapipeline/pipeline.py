from typing import Dict, List, Callable, Union
from datasets import DatasetDict
from tqdm import tqdm
import torch
import json
import os

from .processors import ProcessorFactory
from .packing import SequencePacker
from .io import IOHandler
from .text import TextNormalizer


class DataPipeline:
    """数据处理管道 - 模板方法模式"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "dataset")
    
    def process_dataset(
        self,
        dataset_dict: DatasetDict,
        output_subdir: str,
        max_chunk_num: int = None,
        chunk_size: int = 1000000,
        split_name: str = "train",
        column_name: str = "text",
        process_func: Union[Callable[[dict], dict], Callable[[List[dict]], List[dict]]] = None,
        normalization_func: Callable[[str], str] = None,
        output_dir: str = None,
    ) -> None:
        """处理数据集的主流程"""
        dataset = dataset_dict[split_name]
        total_samples = len(dataset)
        num_chunks = (total_samples // chunk_size) + 1
        lim_chunks = min(max_chunk_num, num_chunks) if max_chunk_num else num_chunks
        
        output_dir = output_dir or os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个数据块
        for i in range(lim_chunks):
            self._process_chunk(
                dataset=dataset,
                chunk_idx=i,
                chunk_size=chunk_size,
                total_samples=total_samples,
                output_dir=output_dir,
                output_subdir=output_subdir,
                column_name=column_name,
                process_func=process_func,
                normalization_func=normalization_func
            )
    
    def _process_chunk(
        self,
        dataset,
        chunk_idx: int,
        chunk_size: int,
        total_samples: int,
        output_dir: str,
        output_subdir: str,
        column_name: str,
        process_func,
        normalization_func
    ) -> None:
        """处理单个数据块"""
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        chunk = dataset.select(range(start_idx, end_idx))
        
        output_path = os.path.join(output_dir, f"{output_subdir}_text_chunk_{chunk_idx}.jsonl")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                processed = self._process_example(
                    example, column_name, process_func, normalization_func
                )
                self._write_processed(f, processed)
        
        print(f"Saved text chunk {chunk_idx} to {output_path}")
    
    def _process_example(
        self,
        example: dict,
        column_name: str,
        process_func,
        normalization_func
    ) -> Union[dict, List[dict]]:
        """处理单个样本"""
        if process_func:
            return process_func(example)
        
        text = example[column_name]
        if normalization_func:
            text = normalization_func(text)
        return {column_name: text}
    
    def _write_processed(self, file, processed) -> None:
        """写入处理后的数据"""
        if isinstance(processed, dict):
            file.write(json.dumps(processed, ensure_ascii=False) + "\n")
        elif isinstance(processed, list):
            for item in processed:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def cache_files(
        self,
        tokenizer,
        files: List[str],
        base_out_dir: str,
        cache_type: str,
        packing_size: int = -1,
        pad_value: int = 1
    ) -> None:
        """缓存文件到H5格式"""
        processor = ProcessorFactory.create(cache_type, tokenizer)
        self.dump_files(
            files=files,
            base_out_dir=base_out_dir,
            process_func=processor.process,
            output_keys=processor.output_keys,
            packing_size=packing_size,
            pad_value=pad_value
        )
    
    def dump_files(
        self,
        files: List[str],
        base_out_dir: str,
        process_func: Callable[[dict], dict],
        output_keys: List[str],
        packing_size: int = -1,
        pad_value: int = 0
    ) -> None:
        """转储文件到H5格式"""
        for file_path in files:
            self._dump_single_file(
                file_path=file_path,
                base_out_dir=base_out_dir,
                process_func=process_func,
                output_keys=output_keys,
                packing_size=packing_size,
                pad_value=pad_value
            )
    
    def _dump_single_file(
        self,
        file_path: str,
        base_out_dir: str,
        process_func: Callable[[dict], dict],
        output_keys: List[str],
        packing_size: int,
        pad_value: int
    ) -> None:
        """转储单个文件"""
        os.makedirs(base_out_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        out_file_name = file_name.split(".")[0]
        
        # 读取和处理数据
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        arrows: List[Dict[str, torch.Tensor]] = []
        for line in tqdm(lines, desc=f"Processing {file_name}", leave=False):
            line_dict = json.loads(line)
            arrow = process_func(line_dict)
            if arrow is not None:
                arrows.append(arrow)
        
        # 组织输出数据
        package: Dict[str, List[torch.Tensor]] = {
            key: [arrow[key] for arrow in arrows]
            for key in output_keys
        }
        
        # 打包序列（如果需要）
        output_package = {}
        for key in output_keys:
            if packing_size > 0:
                print(f"Packaging key: '{key}'")
                packer = SequencePacker(packing_size, pad_value)
                output_package[key] = packer.pack(package[key])
            else:
                output_package[key] = package[key]
        
        # 保存到H5
        IOHandler.save_h5(base_out_dir, out_file_name, output_package)


# 向后兼容的函数接口
def process_dataset(
    dataset_dict: DatasetDict,
    output_subdir: str,
    max_chunk_num: int = None,
    chunk_size: int = 1000000,
    split_name: str = "train",
    column_name: str = "text",
    process_func: Union[Callable[[dict], dict], Callable[[List[dict]], List[dict]]] = None,
    normalization_func: Callable[[str], str] = None,
    output_dir: str = None,
) -> None:
    """向后兼容的函数接口"""
    pipeline = DataPipeline(output_dir)
    normalizer = TextNormalizer() if normalization_func is None else None
    norm_func = normalization_func or (normalizer.normalize if normalizer else None)
    
    return pipeline.process_dataset(
        dataset_dict=dataset_dict,
        output_subdir=output_subdir,
        max_chunk_num=max_chunk_num,
        chunk_size=chunk_size,
        split_name=split_name,
        column_name=column_name,
        process_func=process_func,
        normalization_func=norm_func,
        output_dir=output_dir
    )


def cache_files(tokenizer, files, base_out_dir, cache_type, packing_size: int = -1, pad_value: int = 1):
    """向后兼容的函数接口"""
    pipeline = DataPipeline()
    return pipeline.cache_files(
        tokenizer=tokenizer,
        files=files,
        base_out_dir=base_out_dir,
        cache_type=cache_type,
        packing_size=packing_size,
        pad_value=pad_value
    )


def dump_files(
    files: List[str],
    base_out_dir: str,
    process_func: Callable[[dict], dict],
    output_keys: List[str],
    packing_size: int = -1,
    pad_value: int = 0
):
    """向后兼容的函数接口"""
    pipeline = DataPipeline()
    return pipeline.dump_files(
        files=files,
        base_out_dir=base_out_dir,
        process_func=process_func,
        output_keys=output_keys,
        packing_size=packing_size,
        pad_value=pad_value
    )
