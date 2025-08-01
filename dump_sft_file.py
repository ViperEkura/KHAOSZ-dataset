from utils import dump_pkl_files, fetch_files
from tokenizer import BpeTokenizer
import torch
import os

def get_processor(tokenizer: BpeTokenizer):
    def processor(input_dict: dict):
        query, response = input_dict["query"], input_dict["response"]
        prefix_seg = f"<|user|> {query} <|system|> <bos>"
        suffix_seg = f"{response}<eos>\n"
        prefix_ids = tokenizer.encode(prefix_seg)
        suffix_ids = tokenizer.encode(suffix_seg)
        
        tokens = prefix_ids + suffix_ids
        tokens = torch.tensor(tokens, dtype=torch.int32)
        masks = torch.zeros_like(tokens, dtype=torch.bool)
        masks[len(prefix_ids):] = True
        
        return {"sequence": tokens, "mask": masks}
    
    return processor


if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = [
        os.path.join("dataset", "Ling-Coder-SFT"),
        os.path.join("dataset", "chinese-instruct")
    ]
    base_out_dir = "pkl_output"
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
        
    processor = get_processor(tokenizer)
    
    dump_pkl_files(files, base_out_dir,processor, ["sequence", "mask"])