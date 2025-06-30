from datasets import load_dataset
import json
import os

if __name__ == "__main__":
    dataset_dict = load_dataset("shjwudp/chinese-c4")
    train_dataset = dataset_dict["train"]

    chunk_size = 1000000
    total_samples = len(train_dataset)
    num_chunks = (total_samples // chunk_size) + 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path(script_dir, "dataset", "chinese-c4")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk = train_dataset.select(range(start_idx, end_idx))

        output_path = f"{output_dir}/{output_dir}_text_chunk_{i}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in chunk:
                # 每行写入一个 {"text": "xxx"} 对象
                json_line = {"text": example["text"]}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(f"Saved text chunk {i} to {output_path}")
