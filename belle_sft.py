from datasets import load_dataset
from utils import process_dataset

if __name__ == "__main__":
    dataset = load_dataset("BelleGroup/train_3.5M_CN")
    # process_dataset(
    #     dataset_dict=dataset,
    #     output_subdir="belle_sft",
    #     max_chunk_size=5,
    # )