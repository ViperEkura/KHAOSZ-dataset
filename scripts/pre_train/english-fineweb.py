from datasets import load_dataset
from pipeline import export_dataset

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="english-fineweb-pretrain",
    )
