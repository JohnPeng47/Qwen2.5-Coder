import json
import sacrebleu
from client import VLLMClient

NUM_COMPLETES = 10
NUM_SAMPLES = 4
CODE_QA_DATA = "train/gpt-4o_5_file_noreason_train.json"
CODE_COMPLETE_TEMPLATE = """
Given the following code from the repo Aider, complete the rest:

----------- CODE -------------
{code}
"""
client = VLLMClient()

def calculate_bleu(completion, reference):
    # SacreBLEU expects a list of references for each sentence
    references = [[reference]]
    # The completion should be a single string
    hypothesis = [completion]
    
    bleu = sacrebleu.corpus_bleu(hypothesis, references)
    return bleu.score

if __name__ == "__main__":
    with open(CODE_QA_DATA) as f:
        data = json.load(f)

    completed = 0
    total_bleu = 0
    
    for i in range(0, len(data), 5):
        item = data[i]
        q = item["messages"][0]["content"]
        a = item["messages"][1]["content"]
        if "Aider" in a:
            a = a.split("\n")
            if len(a) < 50:
                continue

            prefix, reference = a[10:30], a[30:50]            
            prompt = CODE_COMPLETE_TEMPLATE.format(
                code = "\n".join(prefix)
            )
            
            sample_scores = []
            for _ in range(NUM_SAMPLES):
                completion = client.generate(prompt)["choices"][0]["text"]
                reference_text = "\n".join(reference)
                bleu_score = calculate_bleu(completion, reference_text)
                sample_scores.append(bleu_score)
            
            avg_sample_score = sum(sample_scores) / len(sample_scores)
            max_sample_score = max(sample_scores)
            total_bleu += avg_sample_score
            
            print(f"Sample {completed + 1}:")
            print(f"Average BLEU Score: {avg_sample_score:.2f}")
            print(f"Highest BLEU Score: {max_sample_score:.2f}")
            print("-" * 50)
                        
            completed += 1

        if NUM_COMPLETES == completed:
            break
    
    # Calculate and print average BLEU score
    if completed > 0:
        avg_bleu = total_bleu / completed
        print(f"\nAverage BLEU Score across {completed} samples: {avg_bleu:.2f}")