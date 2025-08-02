# inclusionAI/Ling-Coder-SFT
from datasets import load_dataset
from modules.utils import process_dataset


def process_func(input_dict: dict) -> dict:
    msg = input_dict["messages"]
    query = msg[0]["content"]
    history = msg[1]["content"]
    return {"query": query, "response": history}


if __name__ == "__main__":
    dataset = load_dataset("inclusionAI/Ling-Coder-SFT")
    
    process_dataset(
        dataset_dict=dataset,
        output_subdir="Ling-Coder-sft",
        process_func=process_func
    )
    