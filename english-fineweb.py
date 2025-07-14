from utils import process_dataset
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
    process_dataset(
        dataset_name="HuggingFaceFW/fineweb",
        output_subdir="english-fineweb",
        dataset_config="sample-10BT",
        normalization_func=comprehensive_normalization
    )