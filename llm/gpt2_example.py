import os
proxy = 'http://127.0.0.1:7890'

proxies = {
    'http': proxy,
    'https': proxy
}

os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# print(tokenizer)

model = AutoModelForCausalLM.from_pretrained("gpt2")

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

