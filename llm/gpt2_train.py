import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_line_length = 400

# 读取训练数据集
with open("train.txt", "r", encoding="utf-8") as file:
    data = file.read().split('\n')
    texts = []
    current = ''
    for line in data:
        line = line.strip()
        if line != '':
            current += line
            if len(current) > max_line_length:
                pre = current[:max_line_length]
                texts.append(pre)
                current = current[max_line_length:]
    if len(current) > 0:
        texts.append(current)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, text):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text = text

    def __getitem__(self, index):
        text = self.text[index]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.squeeze(0).to(device)
        input_index = input_ids[:-1]
        label = input_ids[1:]
        return {
            'input_index': input_index,
            'label':  label
        }

    def __len__(self):
        return len(self.text)


class OptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        print('Model loaded from pretrained gpt2:')
        print(self.model)
        print("---------------------------------------")

    def forward(self, input_index, label):
        outputs = self.model(input_index, labels=label)
        # 'outputs' is a tuple, we need the first element which are the logits
        lm_logits = outputs.logits
        if label is not None:
            loss = self.loss_func(lm_logits.view(-1, lm_logits.size(-1)), label.view(-1))
            return loss
        else:
            pred = torch.argmax(lm_logits, dim=-1)
            return pred


def process(batch_data):
    input_index = torch.nn.utils.rnn.pad_sequence([item['input_index'] for item in batch_data],
                                                  batch_first=True,
                                                  padding_value=0)
    label = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch_data],
                                            batch_first=True,
                                            padding_value=0)
    input_index = input_index.to(device)
    label = label.to(device)
    return {
        'input_index': input_index,
        # 'input_attention_mask': input_attention_mask,
        'label': label
    }


batch_size = 12
epoch = 64

print('Start training...')
print('Device:', device)
print('Batch size:', batch_size)
print('Epoch:', epoch)

train_dataset = MyDataset(texts)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=process)

model = OptModel().to(device)
model = model.train()

optim = torch.optim.Adam(model.parameters(), lr=1e-5)


for e in range(epoch):
    # Training loop
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc='Training', position=0, leave=True)
    for data in progress_bar:
        data['input_index'] = data['input_index'].to(device)
        data['label'] = data['label'].to(device)

        optim.zero_grad()
        loss = model(data['input_index'], data['label'])
        loss.backward()
        optim.step()

        train_loss += loss.item()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(data['input_index']))})
    avg_train_loss = train_loss / len(train_dataloader)
    print(f'Average training loss: {avg_train_loss}')

# Save the model
torch.save(model.state_dict(), 'gpt2_model.pth')
print('Model saved to gpt2_model.pth')
