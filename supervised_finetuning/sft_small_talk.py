# opencsg/smoltalk-chinese
from datasets import load_dataset
from modules.utils import process_dataset


def process_func(input_dict: dict):
    conversations = input_dict["conversations"]
    assert len(conversations) % 2 == 0
    n = len(conversations) // 2
    examples = []
    
    for i in range(n):
        user_msg = conversations[2*i]["value"]
        assistant_msg = conversations[2*i+1]["value"]
        examples.append({
            "query": user_msg,
            "response": assistant_msg
        })
    
    return examples


if __name__ == "__main__":
    dataset = load_dataset("opencsg/smoltalk-chinese")
    process_dataset(
        dataset_dict=dataset,
        output_subdir="Magpie-Pro-300K-sft",
        process_func=process_func,
        split_name="train_sft",
    )