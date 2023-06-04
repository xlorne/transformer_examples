import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.nn import Transformer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 数据读取
data = pd.read_csv('data.csv')
data = data.drop(columns=['|', 'Date'])  # 删除无关列

# 选择特征
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Direction', 'Amplitude']
target = 'Close'  # 预测目标
seq_length = 60  # 序列长度

# 数据归一化
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 序列截断
X = np.array([data[features].values[i: i + seq_length] for i in range(len(data) - seq_length - 1)])  # 减去 1，留出目标序列
y = np.array([data[target].values[i + seq_length] for i in range(len(data) - seq_length - 1)])  # 目标序列

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据转化为 PyTorch 的 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 模型定义
class StockPredictor(nn.Module):
    def __init__(self, feature_size, num_heads, num_layers, dropout):
        super(StockPredictor, self).__init__()
        self.transformer = Transformer(d_model=feature_size, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(feature_size, 1)

    def forward(self, x):
        x = self.transformer(x, x)  # 使用相同的输入作为目标序列
        x = self.fc(x[:, -1, :])
        return x

feature_size = len(features)
num_heads = feature_size  # 设置 num_heads 为 feature_size
num_layers = 3
dropout = 0.2
model = StockPredictor(feature_size, num_heads, num_layers, dropout)
model.to(device)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 模型训练
epochs = 100
for epoch in range(epochs):
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 模型评估
model.eval()
y_pred_test = model(X_test)
test_loss = loss_fn(y_pred_test.squeeze(), y_test)
print(f'Test Loss: {test_loss.item()}')
