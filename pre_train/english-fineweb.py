from datasets import load_dataset
from modules.datapipeline import DataPipeline

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="english-fineweb-pretrain",
    )