from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/code/lorne/models/vicuna-13b-v1.3")

# print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("/code/lorne/models/vicuna-13b-v1.3")

input_context = "The weather is really nice. " \
                "The sky is clear. I am very happy today. " \
                "I am going to go to the park to play."
input_ids = tokenizer.encode(input_context, return_tensors='pt')
#
# dicts = tokenizer.get_vocab()
# print(dicts)
# print(input_ids)

print(model)

# Generate 50 tokens
output = model.generate(input_ids, max_length=50)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

