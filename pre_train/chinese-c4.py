from datasets import load_dataset
from modules.datapipeline import DataPipeline

if __name__ == "__main__":
    dataset = load_dataset("shjwudp/chinese-c4")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="chinese-c4-pretrain"
    )