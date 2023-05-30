import torch
import torch.nn as nn
import torch.optim as optim

# 参数定义
input_vocab_size = 1000
output_vocab_size = 1000
d_model = 512

# Transformer参数
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048

# 词向量的参数
max_seq_length = 20
batch_size = 64

# 训练次数
num_epochs = 100


class SimpleTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward):
        super(SimpleTransformer, self).__init__()

        # 是 PyTorch 中的一个模块，用于实现词嵌入（word embedding）。词嵌入是将离散的词汇（通常表示为 one-hot 向量）映射到连续的向量空间中的低维向量表示。
        # 这种向量表示可以捕捉词汇之间的语义关系，如相似性、类比等。在自然语言处理任务中，词嵌入通常作为神经网络的输入层来使用。
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)

        # Transformer 是一个序列到序列的模型，它由编码器（encoder）和解码器（decoder）组成。
        # 编码器将一个可变长度的输入序列（source）映射为一个定长的连续表示（context vector），
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

        # 解码器将该连续表示（context vector）映射为一个可变长度的输出序列（target）。
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src)
        tgt = self.decoder_embedding(tgt)

        output = self.transformer(src, tgt)

        return self.fc_out(output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleTransformer(input_vocab_size, output_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                          dim_feedforward).to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建模拟数据
src = torch.randint(0, input_vocab_size, (max_seq_length, batch_size)).to(device)
tgt_input = torch.randint(0, output_vocab_size, (max_seq_length, batch_size)).to(device)

tgt_output = torch.cat((tgt_input[1:], torch.zeros(1, batch_size, dtype=torch.long, device=device)), dim=0).to(
    device)

# 训练
for epoch in range(num_epochs):
    optimizer.zero_grad()

    predictions = model(src, tgt_input)
    loss = criterion(predictions.reshape(-1, predictions.size(-1)), tgt_output.reshape(-1))
    loss.backward()
    optimizer.step()

    print("Epoch: {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))


