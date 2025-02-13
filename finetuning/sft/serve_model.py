#!/usr/bin/env python3
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import argparse
import os
import torch

def merge_peft_model(model_name: str, output_dir: str):
    """Download and merge a PEFT model with its base model
    
    Args:
        model_name: Name/path of the PEFT model
        output_dir: Directory to save the merged model
    """
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading PEFT model from {model_name}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    
    # Clear CUDA cache if GPU was used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Download and merge PEFT model")
    parser.add_argument("--model-name", type=str, required=True, 
                       help="Name or path of the PEFT model")
    parser.add_argument("--output-dir", type=str, required=True, default="peft_model",
                       help="Directory to save the merged model")
    parser.add_argument("--force", action="store_true",
                       help="Force remerge model even if output directory exists")
    args = parser.parse_args()

    if args.force and os.path.exists(args.output_dir):
        print("Force merge requested. Removing existing merged model...")
        import shutil
        shutil.rmtree(args.output_dir)

    merge_peft_model(args.model_name, args.output_dir)

if __name__ == "__main__":
    main()