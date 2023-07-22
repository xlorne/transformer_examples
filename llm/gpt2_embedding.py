import torch
from transformers import GPT2Model, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
本示例展示了如何使用 GPT2Model 来获取句子的嵌入向量
词向量的输出不会随着输入的长度而改变，因此我们可以得到词嵌入后的固定的向量

这个向量可以用来计算句子的相似度，也可以用于其他的下游任务，
例如：文本分类、文本聚类、文本生成等

很多人好奇为什么ChatGPT接口有最大tokens限制，而官方提供的网页版本却可以无限的交流？
我的认为他的处理机制是这样的：
1. 首先他会将每次输入的问题和回答都保存到起来，同时也会保存对应文本的Embedding词向量数据。
2. 然后在下一次输入的时候，会将最后一次的输入的词向量数据和当前输入的词向量数据进行拼接，然后再输入到模型中。

这样的话模型就可以做到，无论历史对话再多只要单次回答的输入不超过最大tokens的限制了，就可以无限的聊下去，而且无论历史对话再多其实都是一次推理调用，不像接口一样通过累加到history中传递，累加history会随着会话内容的增加，必然会导致触发最大tokens的限制问题。

我的理论支持点：
1. 在ChatGPT的聊天测试过程中，无论交流多长只要不是一个问题的输入内容超过了最大tokens的限制，他都可以无限的回答下去。
2. ChatGPT允许在任何一次历史的会话中编辑问题重新获取答案，这也说明了他不仅保留了每次的历史问答数据，还保留了对应的词向量数据。
"""


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

