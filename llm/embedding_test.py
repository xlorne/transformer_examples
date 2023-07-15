import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
from random import randint

# 定义超参数
input_length = 5  # 输入序列长度
vocab_size = 6  # 词汇表大小
embedding_dim = 10  # 嵌入维度
hidden_dim = 20  # 隐藏层维度
output_dim = 1  # 输出维度
learning_rate = 0.001
num_epochs = 1000

# 定义数据
input_seq = torch.tensor([randint(0, vocab_size-1) for _ in range(input_length)])  # 输入序列
target = torch.tensor([0], dtype=torch.long)  # 目标值

# 定义嵌入层
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_seq):
        embedded = self.embedding_matrix[input_seq]
        return embedded

# 定义嵌入层网络
class EmbeddingNet(nn.Module):
    def __init__(self,input_length, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        kernel_size = 3  # 卷积核大小
        self.conv1d = nn.Conv1d(input_length, hidden_dim, kernel_size)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 输出层的激活函数

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        embedded = self.conv1d(embedded)
        embedded, _ = torch.max(embedded, dim=1)  # 进行最大池化
        hidden = self.fc1(embedded)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
        output = self.sigmoid(output)  # 输出层的激活函数
        return output

# 创建嵌入层网络实例
model = EmbeddingNet(input_length,vocab_size, embedding_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类问题使用二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练网络
with tqdm(total=num_epochs, desc="Epoch", file=sys.stdout) as pbar:
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch + 1}")
        pbar.set_postfix(loss=loss.item())
        pbar.update()

# 使用网络进行预测
with torch.no_grad():
    output = model(input_seq)
    predicted_class = torch.round(output).item()  # 四舍五入为最接近的整数
    print(f"Predicted class: {predicted_class}")
