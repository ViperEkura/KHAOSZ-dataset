from datasets import DatasetDict
from datasets import load_dataset, concatenate_datasets
from utils import process_dataset


def build_prompt(query:str, response:str) -> str:
    replacements = {
        "\\[": "$$",
        "\\]": "$$",
        "\\(": "$",
        "\\)": "$"
    }

    for old, new in replacements.items():
        query = query.replace(old, new)
        response = response.replace(old, new)
         
    return f"<|user|> {query} <|system|> <bos>{response}<eos>\n"


def process_func(input_dict: dict):
    query = input_dict["prompt"]
    response = input_dict["response"]
    return {"text": build_prompt(query, response)}


if __name__ == "__main__":
    all_data = ['stem_zh', 'infinity-instruct', 'firefly', 'magpie', 'dpsk-r1-distil', 
                'coig-cqia', 'disc-law', 'neo_sft_phase2', 'chinese-medical', 'chinese-reasoning-distil', 
                'psycho-10k-dpsk-r1', 'sof-c-zh', 'industryinstruction', 'Chinese-QA-AFAF']
    
    datasets = []
    for subset in all_data:
        ds = load_dataset("Mxode/Chinese-Instruct", name=subset)
        datasets.append(ds["train"]) 
    
    combined_dataset = concatenate_datasets(datasets)
    
    process_dataset(
        dataset_dict=DatasetDict({"train": combined_dataset}),
        output_subdir="chinese-instruct",
        process_func=process_func,
    )