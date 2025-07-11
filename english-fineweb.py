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

if __name__ == "__main__":
    dataset_dict = load_dataset("HuggingFaceFW/fineweb","sample-10BT")
    train_dataset = dataset_dict["train"]

    chunk_size = 1000000
    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path(script_dir, "dataset", "english-fineweb")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        if i == 10: 
            break
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))
        
        output_path = f"{output_dir}/english-fineweb_text_chunk_{i}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                json_line = {"text": comprehensive_normalization(example["text"])}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        
        print(f"Saved text chunk {i} to {output_path}")