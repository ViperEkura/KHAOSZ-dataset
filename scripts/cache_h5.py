"""JSONL → H5 缓存脚本

将 dataset/ 下的 JSONL 文件 tokenize 并打包为 HDF5 格式。

用法:
    python scripts/cache_h5.py pt ./dataset/chinese-c4-pretrain
    python scripts/cache_h5.py sft ./dataset/belle-sft --pack-size 4096
    python scripts/cache_h5.py sft ./dataset/Ling-Coder-sft --tokenizer ./my_tokenizer.json
"""
import argparse
import os

from pipeline import BpeTokenizer, ProcessorFactory, cache_jsonl, IOHandler


def collect_jsonl_files(input_dir: str) -> list[str]:
    """收集目录下所有 JSONL 文件"""
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(input_dir)
        for f in filenames
        if f.endswith(".jsonl")
    ]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="JSONL → H5 缓存")
    parser.add_argument("type", choices=["pt", "sft", "dpo"], help="处理器类型")
    parser.add_argument("input_dir", help="JSONL 文件所在目录")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="H5 输出目录 (默认: <input_dir>/cached)")
    parser.add_argument("-t", "--tokenizer", default="./tokenizer.json",
                        help="Tokenizer 路径 (默认: ./tokenizer.json)")
    parser.add_argument("-p", "--pack-size", type=int, default=-1,
                        help="序列打包长度，<=0 不打包 (默认: -1)")
    parser.add_argument("--pad-value", type=int, default=1,
                        help="打包填充值 (默认: 1 即 <eos>)")
    args = parser.parse_args()

    # 收集 JSONL 文件
    jsonl_files = collect_jsonl_files(args.input_dir)
    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f}")

    # 加载 tokenizer
    if not os.path.exists(args.tokenizer):
        print(f"[ERROR] Tokenizer not found: {args.tokenizer}")
        return
    tokenizer = BpeTokenizer(args.tokenizer)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    # 创建处理器
    processor = ProcessorFactory.create(args.type, tokenizer)
    print(f"Processor: {args.type} ({processor.__class__.__name__})")
    print(f"Output keys: {processor.output_keys}")

    # 输出目录
    output_dir = args.output_dir or os.path.join(args.input_dir, "cached")

    # 执行缓存
    print(f"\nStart caching...")
    if args.pack_size > 0:
        print(f"  pack_size={args.pack_size}, pad_value={args.pad_value}")
    else:
        print(f"  no packing")

    output_files = cache_jsonl(
        files=jsonl_files,
        output_dir=output_dir,
        processor=processor,
        pack_size=args.pack_size,
        pad_value=args.pad_value,
    )

    print(f"\nDone! {len(output_files)} H5 files saved to {output_dir}")


if __name__ == "__main__":
    main()
