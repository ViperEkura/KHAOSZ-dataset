from datasets import load_dataset
from pipeline import export_dataset

if __name__ == "__main__":
    dataset = load_dataset("shjwudp/chinese-c4")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="chinese-c4-pretrain",
    )
