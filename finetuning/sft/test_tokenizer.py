from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("JonhTheTrueKingoftheNorth/SingleRepo_Aider")

print(tokenizer.chat_template)
print(tokenizer.get_chat_template())
