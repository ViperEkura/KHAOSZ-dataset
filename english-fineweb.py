from utils import process_dataset


if __name__ == "__main__":
    process_dataset(
        dataset_name="HuggingFaceFW/fineweb",
        output_subdir="english-fineweb",
        dataset_config="sample-10BT",
    )