from datasets import load_dataset
from utils import process_dataset

if __name__ == "__main__":
    max_chunk_num = 10
    chunk_size = 1000000
    
    dataset = load_dataset(
        "opencsg/chinese-cosmopedia", 
        data_files={"train": [f"data/0000{i}.parquet" for i in range(5)]}
    )
    
    process_dataset(
        dataset_dict=dataset,
        output_subdir="chinese-wiki",
        max_chunk_num=max_chunk_num,
        chunk_size=chunk_size,
    )