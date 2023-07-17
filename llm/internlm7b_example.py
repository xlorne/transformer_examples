from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/code/lorne/models/internlm-chat-7b",trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("/code/lorne/models/internlm-chat-7b",trust_remote_code=True).cuda()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
