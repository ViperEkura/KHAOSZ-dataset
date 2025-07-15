from utils import process_dataset

if __name__ == "__main__":
    max_chunk_num = 10
    chunk_size = 1000000
    item_size = max_chunk_num * chunk_size
    
    process_dataset(
        dataset_name="opencsg/chinese-cosmopedia",
        output_subdir="chinese-wiki",
        data_files=[f"0000{i}.parquet" for i in range(5)],
        max_chunk_num=max_chunk_num,
        chunk_size=chunk_size,
    )