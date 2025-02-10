import json
import argparse
from pathlib import Path
from typing import Optional
import sys
import os
import random
from datasets import load_dataset, Dataset

MIX_DATASET = "HuggingFaceFW/fineweb-edu"
MIX_RATIO = 10              # suggested by Teor

def load_dataset_by_size(
    dataset_name: str,
    target_size_bytes: int,
    split: str = "train",
    streaming: bool = True,
    seed: Optional[int] = None,
    sample_size: int = 100,
) -> Dataset:
    """
    Load a portion of a dataset that approximately matches a target size in bytes.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        target_size_bytes: Desired size in bytes
        split: Dataset split to load from
        streaming: Whether to use streaming mode (recommended for large datasets)
        seed: Random seed for sampling
        sample_size: Number of items to sample for size estimation
    
    Returns:
        Dataset containing approximately target_size_bytes worth of data
    """
    if seed is not None:
        random.seed(seed)
    
    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Sample some items to estimate average size
    sample_items = []
    for item in dataset.take(sample_size):
        sample_items.append(item)
    
    # Calculate average item size
    total_size = 0
    for item in sample_items:
        # Convert item to JSON string to get approximate size in bytes
        item_size = sys.getsizeof(json.dumps(item))
        total_size += item_size
    
    avg_item_size = total_size / len(sample_items)
    
    # Calculate how many items we need
    target_items = int(target_size_bytes / avg_item_size)
    
    # Load the calculated number of items
    if streaming:
        final_dataset = dataset.take(target_items)
    else:
        dataset = load_dataset(dataset_name, split=f"{split}[:{target_items}]")
        final_dataset = dataset
    
    actual_size = sum(sys.getsizeof(json.dumps(item)) for item in final_dataset)
    print(f"Requested size: {target_size_bytes:,} bytes")
    print(f"Actual size: {actual_size:,} bytes")
    print(f"Number of items: {target_items:,}")
    
    return final_dataset

def create_conversation_pair(question, code_content):
    """Create a single Q&A pair with code context"""
    instruction = f"Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\n{question}"
    response = f"Let me analyze the code and answer your question about it.\n\n{code_content}"
    
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]

def create_training_data(sample):
    """Convert sample into training conversations"""
    instruct_ft_data = []
    code_size = 0
    for code_block in sample["code_blocks"]:
        for question in code_block["questions"]:
            conversation = create_conversation_pair(
                question=question,
                code_content=code_block["content"]
            )
            instruct_ft_data.extend(conversation)

        code_size += len(code_block["content"])
    
    instruct_ft_data += load_dataset_by_size(MIX_DATASET, code_size * MIX_RATIO)
    return instruct_ft_data

if __name__ == "__main__":
    # Check that current working directory ends with finetuning/sft
    cwd = Path.cwd()
    if not str(cwd).endswith(f"finetuning{os.sep}sft"):
        raise ValueError("Script must be run from finetuning/sft directory")
    
    parser = argparse.ArgumentParser(description="Convert QA data to ChatML format")
    parser.add_argument("data_path", type=str, help="Path to the QA data file")
    parser.add_argument("outdir", type=str, help="Path to output directory")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Load QA data
    qa_data = json.load(open(data_path))

    # Convert to ChatML format
    conversations = create_training_data(qa_data)
    conversations = {
        "messages": conversations
    }

    outfile = outdir / f"{data_path.stem}_train.json"
    with open(outfile, "w") as f:
        f.write(json.dumps(conversations))

    print(f"Converted {data_path} to ChatML format")
    print(f"Saved to {outfile}")
