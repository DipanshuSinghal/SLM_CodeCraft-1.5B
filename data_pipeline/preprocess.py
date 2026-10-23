import os
from datasets import load_dataset, DatasetDict

def format_chatml_prompt(example: dict) -> dict:
    """Format instruction-response pairs into standardized ChatML format."""
    instruction = example.get("instruction", "")
    input_ctx = example.get("input", "")
    output = example.get("output", "")

    if input_ctx:
        user_content = f"{instruction}\n\nContext/Input:\n{input_ctx}"
    else:
        user_content = instruction

    formatted_text = (
        f"<|im_start|>system\nYou are CodeCraft-1.5B, an expert Python code synthesis assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    
    return {"text": formatted_text}

def prepare_dataset(dataset_name: str = "sahil2801/CodeAlpaca-20k", output_dir: str = "./data_pipeline/processed"):
    """Load, format, and save dataset locally."""
    print(f"Fetching dataset: {dataset_name}...")
    raw_dataset = load_dataset(dataset_name)

    print("Formatting prompts to ChatML structure...")
    processed_dataset = raw_dataset.map(
        format_chatml_prompt,
        remove_columns=raw_dataset["train"].column_names,
        desc="Formatting dataset"
    )

    os.makedirs(output_dir, exist_ok=True)
    processed_dataset.save_to_disk(output_dir)
    print(f"Dataset saved successfully to '{output_dir}'.")

if __name__ == "__main__":
    prepare_dataset()