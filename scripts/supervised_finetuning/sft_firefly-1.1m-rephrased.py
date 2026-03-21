from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    return {"query": input_dict["instruction"], "response": input_dict["output"]}


if __name__ == "__main__":
    dataset = load_dataset("Mxode/Firefly-1.1M-Rephrased")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="Firefly-1.1M-Rephrased",
        process_func=process_func,
    )
