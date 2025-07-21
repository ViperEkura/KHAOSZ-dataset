from typing import List, Callable, Union
from datasets import DatasetDict
from tokenizer import BpeTokenizer
from tqdm import tqdm
from torch import Tensor

import torch.nn.functional as F
import pickle as pkl
import torch
import json
import os
import re


def fetch_files(directory):
    return [os.path.join(root, f) 
            for root, _, files in os.walk(directory) for f in files]

def comprehensive_normalization(text):
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u0060': "'",
        '\u201C': '"', '\u201D': '"', 
        '\u2013': '-', '\u2014': '--', '\u2212': '-',
        '\u00A0': ' ',
        '\u2026': '...'
    }
    pattern = re.compile('|'.join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group()], text)      

   
def dump_pkl_files(
    tokenizer: BpeTokenizer, 
    files: List[str], 
    base_out_dir: str, 
    encoder: Callable[[str], str]=None,
    key: str='text',
    packing_size: int=None
):  
    def process_line(line: str) -> Tensor:
            line = json.loads(line)[key]
            processed_line = encoder(line) if encoder else line
            ids = tokenizer.encode(processed_line)
            arrow = torch.tensor(ids, dtype=torch.int32)
            return arrow
    
    for file_path in files:
        out_file_name = os.path.basename(file_path).replace(".jsonl", ".pkl")
        out_file_path = os.path.join(base_out_dir, out_file_name)
        file_name = os.path.basename(file_path)
        arrows: List[Tensor] = []
        
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        with open(file_path, "r") as f:    
            lines = f.readlines()
        for line in tqdm(lines, desc=f"Processing {file_name}", leave=False):
            arrow = process_line(line)
            arrows.append(arrow)
        
        if packing_size is None:
            with open(out_file_path, "wb") as f:
                package_tensor = torch.cat(arrows)
                pkl.dump(package_tensor, f)
        else:
            arrows.sort(key=lambda x: x.numel(), reverse=True)
            packages = []
            cur_size = 0
            cur_tensor = torch.tensor([])
            
            for i in tqdm(range(0, len(arrows)), desc=f"Packing {file_name}", leave=False):
                cur_ids = arrows[i]
                if cur_ids.numel() <= packing_size:
                    if cur_ids.numel() + cur_tensor.numel() <= packing_size:
                        cur_size += cur_ids.numel()
                        cur_tensor = torch.cat([cur_tensor, cur_ids])
                    else:
                        cur_tensor = F.pad(
                            cur_tensor, 
                            (0, packing_size - cur_tensor.numel()),
                            'constant',
                            tokenizer.pad_id
                        )
                        packages.append(cur_tensor)
                        cur_tensor = cur_ids
                else:
                    packages.append(cur_ids[:packing_size])
            
        
            with open(out_file_path, "wb") as f:
                package_tensor = torch.cat(packages)
                pkl.dump(package_tensor, f)
            
                    
            
def process_dataset(
    dataset_dict: DatasetDict,
    output_subdir: str,
    max_chunk_num: int = None,
    chunk_size: int = 1000000,
    split_name: str = "train",
    column_name: str = "text",
    process_func: Callable[[Union[dict, List[dict]]], dict] = None,
    normalization_func=comprehensive_normalization,
):
    train_dataset = dataset_dict[split_name]
    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1
    lim_chunks = min(max_chunk_num, num_chunks) if max_chunk_num else num_chunks

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dataset", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(lim_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))

        output_path = os.path.join(output_dir, f"{output_subdir}_text_chunk_{i}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                if process_func is not None:
                    processed_example = process_func(example)
                else:
                    text = example[column_name]
                    if normalization_func:
                        text = normalization_func(text)
                    processed_example = {column_name: text}
                f.write(json.dumps(processed_example, ensure_ascii=False) + "\n")

        print(f"Saved text chunk {i} to {output_path}")
