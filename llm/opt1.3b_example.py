from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").cuda()
model = model.eval()

input_context = "The weather is really nice. " \
                "The sky is clear. I am very happy today. " \
                "I am going to go to the park to play."
input_ids = tokenizer.encode(input_context, return_tensors='pt')
input_ids = input_ids.cuda()
#
# dicts = tokenizer.get_vocab()
# print(dicts)
# print(input_ids)

print(model)

# Generate 50 tokens
output = model.generate(input_ids, max_length=50)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
