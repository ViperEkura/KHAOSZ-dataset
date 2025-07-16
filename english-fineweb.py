from datasets import load_dataset
from utils import process_dataset

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT")
    process_dataset(
        dataset_dict=dataset,
        output_subdir="english-fineweb",
    )