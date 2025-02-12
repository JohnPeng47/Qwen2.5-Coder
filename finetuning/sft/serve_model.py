#!/usr/bin/env python3
from vllm.entrypoints.openai import api_server
from vllm import LLM, SamplingParams
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import argparse
import os
import torch
import uvloop
from vllm.entrypoints.openai.api_server import run_server


# Static paths
STATIC_MODEL_DIR = os.path.expanduser("~/aider_models")  # Use home directory
MERGED_MODEL_PATH = os.path.join(STATIC_MODEL_DIR, "merged_model")

def ensure_model_merged():
    """Ensure we have a merged model available"""
    if os.path.exists(MERGED_MODEL_PATH):
        print(f"Found existing merged model at {MERGED_MODEL_PATH}")
        return MERGED_MODEL_PATH
    
    print(f"No merged model found. Merging PEFT model...")
    os.makedirs(STATIC_MODEL_DIR, exist_ok=True)
    
    # Load and merge model
    model = AutoPeftModelForCausalLM.from_pretrained(
        "JonhTheTrueKingoftheNorth/SingleRepo_Aider",
        device_map="cuda"  # Load on CPU first
    )
    
    tokenizer = AutoTokenizer.from_pretrained("JonhTheTrueKingoftheNorth/SingleRepo_Aider")
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {MERGED_MODEL_PATH}")
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return MERGED_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description='Start VLLM server with merged PEFT model')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run server on')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--force-merge', action='store_true', help='Force remerge model even if it exists')
    args = parser.parse_args()

    # Remove existing merged model if force-merge is specified
    if args.force_merge and os.path.exists(MERGED_MODEL_PATH):
        print("Force merge requested. Removing existing merged model...")
        import shutil
        shutil.rmtree(MERGED_MODEL_PATH)

    # Ensure we have a merged model
    model_path = ensure_model_merged()
    
    print(f"Initializing VLLM with model from {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True  # Needed for Qwen models
    )

    print(f"Starting server on {args.host}:{args.port}")

    args.model = model_path
    uvloop.run(run_server(args))

if __name__ == "__main__":
    main()