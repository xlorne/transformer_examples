import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch import nn
from torch.nn import Transformer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 数据读取
data = pd.read_csv('../data.csv')
data = data.drop(columns=['|', 'Date'])  # 删除无关列


def show(y_test, y_pred_test,title='Stock Prediction Price'):
    # 创建并拟合一个新的scaler用于目标列
    scaler_y = MinMaxScaler()
    scaler_y.fit_transform(data[[target]])
    # 数据可视化
    plt.figure(figsize=(14, 5))
    # 还原目标值
    if y_test is not None:
        y_test_inverse = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
        # y_test = y_test.cpu().numpy().flatten()
        plt.plot(y_test_inverse, color='blue', label='Actual closing price')
    if y_pred_test is not None:
        y_pred_inverse = scaler_y.inverse_transform(y_pred_test.cpu().numpy().reshape(-1, 1)).flatten()
        # y_pred_test = y_pred_test.cpu().numpy().flatten()
        plt.plot(y_pred_inverse, color='red', label='Predicted closing price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# 选择特征
features = ['Open', 'High', 'Low', 'Close', 'Volume']
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
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 数据转化为 PyTorch 的 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

show(y_test, None, title='Stock Price')

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

batch_size = 128  # 根据你的硬件设定合适的批次大小

# 创建 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
num_layers = 2
dropout = 0.2
model = StockPredictor(feature_size, num_heads, num_layers, dropout)
model.to(device)

# print(model)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 模型训练
epochs = 100
with tqdm(total=epochs, desc="Epoch", file=sys.stdout) as pbar:
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        average_loss = total_loss / len(train_loader.dataset)

        pbar.set_postfix(loss="{:.4f}".format(average_loss))
        pbar.update()

# 模型评估
model.eval()

X_test = X_test.to(device)
y_test = y_test.to(device)

with torch.no_grad():
    y_pred_test = model(X_test)

test_loss = loss_fn(y_pred_test.squeeze(), y_test)
print(f'Test Loss: {test_loss.item()}')

show(y_test, y_pred_test)



