from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    conversations = input_dict["conversations"]
    n = len(conversations) // 2
    examples = []
    for i in range(n):
        user_msg = conversations[2 * i]["value"]
        assistant_msg = conversations[2 * i + 1]["value"]
        examples.append({"query": user_msg, "response": assistant_msg})
    return examples


if __name__ == "__main__":
    dataset = load_dataset("BelleGroup/train_3.5M_CN")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="belle-sft",
        process_func=process_func,
    )
