from tokenizer import BpeTokenizer
from tqdm import tqdm
from typing import List
import pickle as pkl
import torch
import os

def fetch_files(directory):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files]

def convert_to_ids(tokenizer: BpeTokenizer, file_path, out_file_path):
    arrows = []
    with open(file_path, "r") as f:
        for line in f:
            ids = tokenizer.encode(line)
            arrows.append(ids)
    
    with open(out_file_path, "w") as f:
        tensor = torch.tensor(arrows, dtype=torch.int32)
        pkl.dump(tensor, f)
        
def process_files(tokenizer: BpeTokenizer, files: List[str], base_out_dir):
    for file_path in tqdm(files, desc="Processing files", total=len(files)):
        out_file_name = os.path.basename(file_path).replace(".jsonl", ".pkl")
        out_file_path = os.path.join(base_out_dir, out_file_name)
        convert_to_ids(tokenizer, file_path, out_file_path)


if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = [
        os.path.join("dataset", "chinese-c4"),
        os.path.join("dataset", "english-fineweb")
    ]
    base_out_dir = "cache"
    
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
    
    process_files(tokenizer, files, base_out_dir)