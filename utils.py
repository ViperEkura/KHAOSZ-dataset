from datasets import load_dataset
import json
import os
import re


def comprehensive_normalization(text):
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u0060': "'",
        '\u201C': '"', '\u201D': '"', 
        '\u2013': '-', '\u2014': '--', '\u2212': '-',
        '\u00A0': ' ',
        '\u2026': '...'
    }
    pattern = re.compile('|'.join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group()], text)      

def process_dataset(
    dataset_name: str,
    output_subdir: str,
    dataset_config: str = None,
    max_chunk_size: int = None,
    split_name: str = "train",
    column_name: str = "text",
    chunk_size: int = 1000000,
    normalization_func=comprehensive_normalization
):

    dataset_dict = load_dataset(dataset_name, dataset_config)
    train_dataset = dataset_dict[split_name]

    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1
    lim_chunks = max_chunk_size if max_chunk_size else num_chunks

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "dataset", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(lim_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))

        output_path = os.path.join(output_dir, f"{output_subdir}_text_chunk_{i}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                text = example[column_name]
                if normalization_func:
                    text = normalization_func(text)
                json_line = {column_name : text}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(f"Saved text chunk {i} to {output_path}")
        
  
