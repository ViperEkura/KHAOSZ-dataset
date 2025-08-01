from datasets import load_dataset
from modules.utils import process_dataset

if __name__ == "__main__":
    dataset = load_dataset("shjwudp/chinese-c4")
    process_dataset(
        dataset_dict=dataset,
        output_subdir="chinese-c4"
    )