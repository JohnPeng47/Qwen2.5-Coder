import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, setup_chat_format
import trl
from datasets import load_dataset, Dataset
import argparse
from huggingface_hub import login
import wandb

from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
)

def get_gpu_memory():
    """Returns GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code QA pairs from a repository")
    parser.add_argument("data_path", type=str, help="Path to the repository to analyze")

    args = parser.parse_args()
    
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dataset = load_dataset("json", data_files=args.data_path)

    # login(token=HF_TOKEN)
    # wandb.login(key=WANDB_KEY)

    # Print initial GPU memory
    print(f"Initial GPU memory usage: {get_gpu_memory():.2f} GB")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    # Print final GPU memory
    print(f"GPU memory after loading model: {get_gpu_memory():.2f} GB")
    print(f"Total GPU memory used: {get_gpu_memory():.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right' # to prevent warnings
    tokenizer.chat_template = None

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    args = SFTConfig(
        # output_dir=".",                         # DONT FORGET TO SET THIS MORRON
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        # learning_rate=2e-4
        learning_rate=1e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # CLAUDE suggest 1.0
        # warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        warmup_ratio=0.03,
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="wandb",                # report metrics to tensorboard
        max_seq_length=8096,                    # Qwen
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,         # We template with special tokens
            "append_concat_token": False,        # No need to add additional separator token
        },
        dataset_text_field="messages"
    )
    
    print("LEN OF DATASET: ", len(dataset))
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"], # THIS FUCKING KEY DOESNT EVEN NEED TO EXIST IN YOUR DATASET !!!!
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model 
    trainer.save_model()