# Mxode/Firefly-1.1M-Rephrased
from datasets import load_dataset
from modules.datapipeline import DataPipeline

def process_func(input_dict: dict):
    instruction = input_dict["instruction"]
    output = input_dict["output"]
    return {"query": instruction, "response": output}


if __name__ == "__main__":
    dataset = load_dataset("Mxode/Firefly-1.1M-Rephrased")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="Firefly-1.1M-Rephrased",
        process_func=process_func
    )
