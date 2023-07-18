import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
model = model.to(device)
model = model.eval()

print(model)

# 输入文本，其中[MASK]表示需要预测的词
input_text = "你好，我是小明，现在是一名程[MASK]员，很高心认[MASK]你，听说你曾经也是[MASK][MASK][MASK]。"

inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

# 模型预测获得结果，这是一个softmax分类结果
logits = model(inputs).logits.squeeze(0)
# print(logits.shape)
# 获取预测结果中概率最大的token id
predicted_class_id = logits.argmax(axis=-1)
# print(predicted_class_id)
# 将token id转化为对应的token，即预测结果
output = tokenizer.decode(predicted_class_id)
# 输出的预测结果与输入文本存在错位，因为训练过程中所采用的自监督任务时，
# 就是将后面的词作为预测目标输出的，所以存在错位
# print('predict output:', output)

# 获取输入文本中[MASK]的位置
mask_token_index = (inputs[0] == tokenizer.mask_token_id).nonzero().squeeze()
# print(mask_token_index)

# 将预测结果替换到输入文本中
for mask_token_index in mask_token_index:
    predicted_token_id = logits[mask_token_index].argmax(axis=-1)
    inputs[0][mask_token_index] = predicted_token_id
# 将token id转化为对应的token，即预测结果
print('\n\n')
output = tokenizer.decode(inputs[0], skip_special_tokens=True).replace(' ', '')
print('input text:', input_text)
print('output text:', output)



