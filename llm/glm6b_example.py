from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/code/lorne/models/chatglm-6b", trust_remote_code=True)

print(tokenizer)

model = AutoModel.from_pretrained("/code/lorne/models/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

print(model)

response, history = model.chat(tokenizer, "你好", history=[])

print(response)
