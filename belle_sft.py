from datasets import load_dataset
from utils import process_dataset


def build_prompt(query, history) -> str:
    ret_prompt = ""
    if len(history) > 0:
        for his_query, his_response in history:
            ret_prompt += f"<|user|> {his_query} <|system|> <bos>{his_response}<eos>\n"
    if query is not None:
        ret_prompt += f"<|user|> {query} <|system|> <bos>"
    return ret_prompt


def process_func(input_dict: dict):
    conversations = input_dict["conversations"]
    n = len(conversations) // 2
    examples = []
    
    for i in range(n):
        user_msg = conversations[2*i]["value"]
        assistant_msg = conversations[2*i+1]["value"]
        examples.append((user_msg, assistant_msg))
    
    content = {
        "text": build_prompt(None, examples)
    }
    print(content)
    return content


if __name__ == "__main__":
    dataset = load_dataset("BelleGroup/train_3.5M_CN")
    process_dataset(
        dataset_dict=dataset,
        output_subdir="belle_sft",
        chunk_size=5,
        process_func=process_func,
    )