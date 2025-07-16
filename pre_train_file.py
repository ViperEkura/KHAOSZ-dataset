from utils import dump_pkl_files, fetch_files
from tokenizer import BpeTokenizer
import os

def processor(intput_str: str):
    return f"{intput_str} <eos>"

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
    
    dump_pkl_files(tokenizer, files, base_out_dir, processor)