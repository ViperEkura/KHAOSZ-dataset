from utils import dump_pkl_files, fetch_files
from tokenizer import BpeTokenizer
import os

if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = [
        # os.path.join("dataset", "belle-sft"),
        os.path.join("dataset", "chinese-instruct")
    ]
    base_out_dir = "pkl_output"
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
    
    dump_pkl_files(tokenizer, files, base_out_dir, packing_size=2048)