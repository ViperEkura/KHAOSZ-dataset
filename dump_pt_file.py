from modules.utils import dump_pkl_files, fetch_files, fetch_folders, get_pt_processor
from modules.tokenizer import BpeTokenizer


if __name__ == "__main__":
    tokenizer = BpeTokenizer("tokenizer.json")
    base_dir = fetch_folders("dataset")
    
    base_out_dir = "pkl_output"
    files = []
    for dir_path in base_dir:
        files.extend(fetch_files(dir_path))
    processor = get_pt_processor(tokenizer)
    dump_pkl_files(files, base_out_dir, processor, ["text"])