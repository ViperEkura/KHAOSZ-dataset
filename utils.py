from typing import Dict, List, Callable, Union
from datasets import DatasetDict
from tqdm import tqdm
from torch import Tensor

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

def pack_sequences(sequences: List[Tensor], pack_size: int, pad_value: int) -> List[Tensor]:
    packages = []
    sequences.sort(key=lambda x: x.numel(), reverse=True)
    current_pack = torch.tensor([], dtype=torch.int32)
    
    for tensor in sequences:
        if tensor.numel() > pack_size:
            packages.append(tensor[:pack_size])
            continue
            
        remaining = pack_size - current_pack.numel()
        
        if remaining == 0:
            packages.append(current_pack)
            current_pack = tensor
        elif tensor.numel() <= remaining:
            current_pack = torch.cat([current_pack, tensor])
        else:
            padding = torch.full((remaining,), pad_value, dtype=torch.int32)
            current_pack = torch.cat([current_pack, padding])
            packages.append(current_pack)
            current_pack = tensor
    
    if current_pack.numel() > 0:
        if current_pack.numel() < pack_size:
            padding = torch.full(
                (pack_size - current_pack.numel(),), 
                pad_value, 
                dtype=torch.int32
            )
            current_pack = torch.cat([current_pack, padding])
        else:
            current_pack = current_pack[:pack_size]
            
        packages.append(current_pack)
        
    return packages

def dump_pkl_files(
    files: List[str], 
    base_out_dir: str,
    process_func: Callable[[dict], dict],
    output_keys: List[str],
    packing_size: int = -1,
    pad_value: int = 0
):
        
    for file_path in files:
        out_file_name = os.path.basename(file_path).replace(".jsonl", ".pkl")
        out_file_path = os.path.join(base_out_dir, out_file_name)
        file_name = os.path.basename(file_path)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        
        arrows: List[Dict[str, Tensor]] = []
        
        with open(file_path, "r") as f:    
            lines = f.readlines()
            
        for line in tqdm(lines, desc=f"Processing {file_name}", leave=False):
            line_dict = json.loads(line)
            arrow = process_func(line_dict)
            arrows.append(arrow)
        
        package: Dict[str, List[Tensor]] = {}
        for key in output_keys:
            list_tensor = [arrow[key] for arrow in arrows]
            package[key] = list_tensor
            
        output_package: Dict[str, Tensor] = {}
        
        for key in output_keys:
            if packing_size > 0:
                print(f"Packaging key: '{key}'")
                package[key] = pack_sequences(package[key], packing_size, pad_value)
            sequence = torch.cat(package[key])
            output_package[key] = sequence
         
        with open(out_file_path, "wb") as f:
            pkl.dump(output_package, f)
            
                    
            
def process_dataset(
    dataset_dict: DatasetDict,
    output_subdir: str,
    max_chunk_num: int = None,
    chunk_size: int = 1000000,
    split_name: str = "train",
    column_name: str = "text",
    process_func: Union[Callable[[dict], dict], Callable[[List[dict]], List[dict]]] = None,
    normalization_func=comprehensive_normalization,
    output_dir: str = None,
):
    train_dataset = dataset_dict[split_name]
    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1
    lim_chunks = min(max_chunk_num, num_chunks) if max_chunk_num else num_chunks

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "dataset", output_subdir)
        
    os.makedirs(output_dir, exist_ok=True)

    for i in range(lim_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))

        output_path = os.path.join(output_dir, f"{output_subdir}_text_chunk_{i}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                if process_func:
                    processed_example = process_func(example)
                else:
                    text = example[column_name]
                    if normalization_func:
                        text = normalization_func(text)
                    processed_example = {column_name: text}
                
                if isinstance(processed_example, dict):
                    f.write(json.dumps(processed_example, ensure_ascii=False) + "\n")
                elif isinstance(processed_example, list):
                    for item in processed_example:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        

        print(f"Saved text chunk {i} to {output_path}")
