import numpy as np
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from torch.utils.data import TensorDataset, DataLoader

# 假设我们有1000天的数据，每天的数据有5个特征(open, close, high, low, volume)
n_days = 1000
n_features = 5
data = np.random.rand(n_days, n_features)

# 数据预处理：这里我们简单地进行归一化处理
data = (data - data.mean(axis=0)) / data.std(axis=0)

# 构造序列数据
n_input_days = 20  # 使用过去10天的数据预测下一天的数据
n_output_days = 1  # 预测下一天的数据
X = np.array([data[i:i+n_input_days] for i in range(n_days - n_input_days - n_output_days + 1)])
Y = np.array([data[i+n_input_days:i+n_input_days+n_output_days, 0] for i in range(n_days - n_input_days - n_output_days + 1)])  # 我们只预测下一天的开盘价

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# 创建 DataLoader
batch_size = 64
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size)


# 定义模型
class TransformerModel(torch.nn.Module):
    def __init__(self, n_features, n_output_days, n_heads=5, n_hid=64):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(n_features, n_heads, n_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
        self.linear = Linear(n_features, n_output_days)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.linear(output.mean(dim=1))
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(n_features, n_output_days).to(device)

# 训练模型
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, loss: {loss.item()}")
