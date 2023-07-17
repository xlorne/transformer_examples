from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

model = model.eval()

print(model)


input = "你好"

input_ids = tokenizer.encode(input, return_tensors='pt')
output = model(input_ids)

print(output)

hidden_states = output.logits
text_representation = hidden_states[0]
print(text_representation)


# Generate 50 tokens
# output = model.generate(input_ids, max_length=50)

