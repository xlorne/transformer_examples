import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

print(model)

max_length = 120

input_context = "你好，我是小明，现在是一名程序员，很高心认识你，听说你曾经也是"

input_ids = tokenizer.encode(input_context, return_tensors='pt').to(device)

output = model.generate(input_ids, max_length=max_length,
                        do_sample=True, top_k=50, top_p=0.95, temperature=0.05)

output_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Find the index of the first period (.) in the generated text
period_index = output_text.find("。")

if period_index != -1:
    output_text = output_text[:period_index+1]  # Include the period in the output
else:
    output_text = output_text[:max_length]  # Use maximum length if no period is found

print(output_text)
