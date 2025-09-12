from datasets import DatasetDict
from datasets import load_dataset, concatenate_datasets
from modules.utils import process_dataset, comprehensive_normalization


def process_func(input_dict: dict):
    query = input_dict["prompt"] if input_dict["prompt"] else ""
    resp = input_dict["response"] if input_dict["response"] else ""
    
    query = comprehensive_normalization(query)
    resp = comprehensive_normalization(resp)
    
    return {"query": query, "response": resp }


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
        output_subdir="chinese-instruct-sft",
        process_func=process_func,
    )