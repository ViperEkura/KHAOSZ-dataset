from datasets import load_dataset
from utils import process_dataset

if __name__ == "__main__":
    dataset = load_dataset("Blaze7451/enwiki_structured_content")
    process_dataset(
        dataset_dict=dataset,
        output_subdir="english-wiki",
        max_chunk_size=5,
    )