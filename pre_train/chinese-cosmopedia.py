from datasets import load_dataset
from modules.utils import process_dataset

if __name__ == "__main__":
    chunk_size = 1000000
    
    dataset = load_dataset(
        "opencsg/chinese-cosmopedia", 
        data_files={"train": [f"data/000{i:02d}.parquet" for i in range(25)]}
    )
    
    process_dataset(
        dataset_dict=dataset,
        output_subdir="chinese-wiki-pretrain",
        chunk_size=chunk_size,
    )