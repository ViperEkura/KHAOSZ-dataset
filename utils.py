from datasets import load_dataset
import json
import os

def process_dataset(
    dataset_name: str,
    output_subdir: str,
    dataset_config: str = None,
    split_name: str = "train",
    chunk_size: int = 1000000,
    normalization_func=None
):

    dataset_dict = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
    train_dataset = dataset_dict[split_name]

    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dataset", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))

        output_path = os.path.join(output_dir, f"{output_subdir}_text_chunk_{i}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                text = example["text"]
                if normalization_func:
                    text = normalization_func(text)
                json_line = {"text": text}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(f"Saved text chunk {i} to {output_path}")
        
