import json
import argparse
from pathlib import Path

def create_conversation(sample):
    chat_ml_template = {
        "messages": []
    }
    for code_block in sample["code_blocks"]:
        print("Codeblock: ", code_block["id"])
        for q in code_block["questions"]:
            chat_ml_template["messages"].append({
                "role": "user",
                "content": q
            })
            chat_ml_template["messages"].append({
                "role": "system",
                "content": code_block["content"]
            })

    return chat_ml_template

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
    chat_data = create_conversation(qa_data)

    # Save ChatML data
    outfile = outdir / f"{data_path.stem}_chat.json"
    json.dump(chat_data, open(outfile, "w"))

    print(f"Converted {data_path} to ChatML format")
    print(f"Saved to {outfile}")
