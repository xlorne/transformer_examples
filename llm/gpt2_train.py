import torch
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取训练数据集
with open("train.txt", "r", encoding="utf-8") as file:
    data = file.read()

texts = data.split("\n")
sentences = [item.strip() for item in texts if item.strip() != '']
print(len(sentences))

# 将文本转换为模型可接受的编码
input_ids = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]
max_length = max(len(ids) for ids in input_ids)
print(max_length)

input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
input_ids = torch.tensor(input_ids)
print(input_ids.shape)

# 将模型设置为训练模式
model.train()
model = model.to(device)

# 将模型包装在DataParallel中
model = DataParallel(model)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练循环
num_epochs = 10
batch_size = 8

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i in range(0, len(input_ids), batch_size):
        batch_inputs = input_ids[i:i+batch_size].to(device)

        # 清除之前计算的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_inputs, labels=batch_inputs)
        loss = outputs.loss
        epoch_loss += loss.item()

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

    # 输出每个epoch的损失
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(input_ids)}")

# 保存训练好的模型
model.module.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
