from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/code/lorne/models/internlm-chat-7b")

# print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("/code/lorne/models/internlm-chat-7b")

input_context = "你好"
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

