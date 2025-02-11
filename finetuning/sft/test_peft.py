from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM

ADAPTER_WEIGHTS = "/workspace/peft-codeqa-Qwen2.5-7b-6-epochs"

# Optional: Configure 4-bit quantization for efficient loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

# Load base model with quantization
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct", 
    trust_remote_code=True
)

# Load your LoRA adapter
model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_WEIGHTS,
    device_map="auto"
)

# Example inference
prompt = "What does repo_map.py file do in Aider?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)