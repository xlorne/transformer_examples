from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").cuda()
model = model.eval()

input_context = "孔明什么时候与刘备结识的？"
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
