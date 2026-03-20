from datasets import load_dataset
from modules.datapipeline import DataPipeline

def process_func(input_dict: dict):
    return {
        "promopt": input_dict["prompt"],
        "chosen": input_dict["chosen"],
        "rejected": input_dict["rejected"]
    }


if __name__ == "__main__":
    dataset = load_dataset("wenbopan/Chinese-dpo-pairs")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="Chinese-dpo-pairs",
        process_func=process_func
    )