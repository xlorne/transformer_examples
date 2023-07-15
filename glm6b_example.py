import os
proxy = 'http://127.0.0.1:7890'

proxies = {
    'http': proxy,
    'https': proxy
}

os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/code/lorne/models/chatglm-6b", trust_remote_code=True)

print(tokenizer)

model = AutoModel.from_pretrained("/code/lorne/models/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

print(model)

response, history = model.chat(tokenizer, "你好", history=[])

print(response)
