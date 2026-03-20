from datasets import load_dataset
from modules.datapipeline import DataPipeline

if __name__ == "__main__":
    dataset = load_dataset("Blaze7451/enwiki_structured_content")
    
    pipeline = DataPipeline()
    pipeline.process_dataset(
        dataset_dict=dataset,
        output_subdir="english-wiki-pretrain",
        max_chunk_num=5,
    )