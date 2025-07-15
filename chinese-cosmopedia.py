from utils import process_dataset

if __name__ == "__main__":
    max_chunk_num = 10
    chunk_size = 1000000
    item_size = max_chunk_num * chunk_size
    
    process_dataset(
        dataset_name="chinese-cosmopedia",
        output_subdir="chinese-wiki",
        split=f"train[:{item_size}]",
        max_chunk_num=max_chunk_num,
        chunk_size=chunk_size,
    )