from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict) -> dict:
    msg = input_dict["messages"]
    query = msg[0]["content"]
    history = msg[1]["content"]
    return {"query": query, "response": history}


if __name__ == "__main__":
    dataset = load_dataset("inclusionAI/Ling-Coder-SFT")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="Ling-Coder-sft",
        process_func=process_func,
    )
