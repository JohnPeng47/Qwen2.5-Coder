import json
import argparse
from pathlib import Path

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
    conversations = []
    
    for code_block in sample["code_blocks"]:
        for question in code_block["questions"]:
            conversation = create_conversation_pair(
                question=question,
                code_content=code_block["content"]
            )
            conversations.extend(conversation)
    
    return conversations


if __name__ == "__main__":
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

    # Save ChatML data as JSONL
    # outfile = outdir / f"{data_path.stem}_train.jsonl"
    # with open(outfile, "w") as f:
    #     for conv in conversations:
    #         f.write(json.dumps(conv) + "\n")

    outfile = outdir / f"{data_path.stem}_train.json"
    with open(outfile, "w") as f:
        f.write(json.dumps(conversations))

    print(f"Converted {data_path} to ChatML format")
    print(f"Saved to {outfile}")
