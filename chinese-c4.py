from datasets import load_dataset
from utils import pre_tarin_process

if __name__ == "__main__":
    dataset = load_dataset("shjwudp/chinese-c4")
    pre_tarin_process(
        dataset_dict=dataset,
        output_subdir="chinese-c4"
    )