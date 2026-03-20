from datasets import load_dataset
from modules.datapipeline import DataPipeline



def process_func(input_dict: dict):
    conversations = input_dict["conversations"]
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
    dataset = load_dataset("BelleGroup/train_3.5M_CN")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="belle-sft",
        process_func=process_func,
    )