import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# 初始化BertTokenizer和BertForQuestionAnswering模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")

# 问答数据集
question = "谁是三国演义中的主角？"
text = "《三国演义》是罗贯中创作的一部长篇历史小说，主要讲述了三国时期的历史故事。其中刘备、关羽、张飞是主要角色。"

# 对问题和文本进行分词和编码
inputs = tokenizer.encode_plus(text,question, add_special_tokens=True, return_tensors="pt")
print(inputs.input_ids.shape)
model.eval()

print(model)

# 获取模型输出并解码得到答案
output = model(**inputs)
print(output)
start_scores = output.start_logits
end_scores = output.end_logits
print(start_scores.shape, end_scores.shape)

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores) + 1
print(start_index, end_index)

# 将编码转换为原始文本中的起始和结束位置
answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index], skip_special_tokens=True)

print("Question:", question)
print("Answer:", answer)
