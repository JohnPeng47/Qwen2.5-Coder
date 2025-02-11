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

TEST_INSTRUCT_QUESTIONS = [
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nWhat is the purpose of the `Analytics` class and how does it manage user consent for data collection and event tracking?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nHow does the `Analytics` class handle errors related to data tracking providers like Posthog and Mixpanel, and what actions does it take when such errors occur?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nHow does the `Analytics` class determine whether to enable or disable analytics based on user consent and the current state of its internal variables?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nHow does the `Analytics` class ensure user privacy when handling UUIDs for data tracking, and what mechanisms are in place to prevent unauthorized data collection?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nHow does the Analytics class utilize system information and user UUIDs to personalize data tracking events while ensuring compliance with user consent preferences?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nWhat are the key differences between the "--gui" and "--browser" options in the given argparse configuration for the aider tool, and how do they affect the user experience when running aider in a graphical interface?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nWhat is the purpose of the `get_parser` function in the `aider\\args.py` file, and how does it contribute to the overall functionality of the `aider` tool?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nWhat role do different model options (e.g., "--model", "--opus", "--sonnet") play in the `aider\\\\args.py` script, and how do they impact AI chat interactions within the tool?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nHow does the argument "--verify-ssl" work in the context of connecting to models, and what are the implications of enabling or disabling SSL verification in the aider tool?'},
    {'role': 'user', 'content': 'Here is a user query about the codebase Aider. You are to locate the block of source code that contains the answer to this question:\nWhat is the significance of the "--model" argument in the `aider\\\\args.py` file, and how does selecting different models (e.g., opus, sonnet, haiku) affect the behavior and output of the AI pair programming tool?'}
]

# Example inference
for q in TEST_INSTRUCT_QUESTIONS:
    prompt = q["content"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Q:", prompt)
    print("A:", response)