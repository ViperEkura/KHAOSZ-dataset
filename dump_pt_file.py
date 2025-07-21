from utils import dump_pkl_files, fetch_files
from tokenizer import BpeTokenizer
import torch
import os

def get_processor(tokenizer: BpeTokenizer):
    def processor(intput_dict: dict) -> dict:
        segment = intput_dict["text"]
        ids = tokenizer.encode(f"{segment} <eos>")
        t_ids = torch.tensor(ids, dtype=torch.int32)
        
        return {'sequence': t_ids}
    
    return processor


if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = [
        os.path.join("dataset", "chinese-c4"),
        os.path.join("dataset", "english-fineweb"),
        os.path.join("dataset", "english-wiki"),
        os.path.join("dataset", "chinese-wiki"),
    ]
    
    base_out_dir = "pkl_output"
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
    processor = get_processor(tokenizer)
    dump_pkl_files(files, base_out_dir, processor, ["text"])