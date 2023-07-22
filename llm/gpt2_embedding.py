import torch
from transformers import GPT2Model, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 输入句子
sentence = "This is an example sentence."

# 分词并转化为张量
inputs = tokenizer(sentence, return_tensors='pt').to(device)

model = model.to(device)
model.eval()

print(model)

# 通过模型的前向传播函数处理输入
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)

# 注意：outputs.last_hidden_state 包含了模型最后一层的输出
sentence_embedding = outputs.last_hidden_state.squeeze(0)

print(sentence_embedding.size())  # 输出：torch.Size([6, 768])
