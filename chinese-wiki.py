from utils import process_dataset

if __name__ == "__main__":
    process_dataset(
        dataset_name="wikimedia/wikipedia",
        output_subdir="chinese-wiki",
        dataset_config="zh"
    )