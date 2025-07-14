from utils import process_files, fetch_files
from tokenizer import BpeTokenizer
import os


if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = [
        os.path.join("dataset", "chinese-c4"),
        os.path.join("dataset", "english-fineweb")
    ]
    base_out_dir = "pkl_output"
    
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
    
    process_files(tokenizer, files, base_out_dir)