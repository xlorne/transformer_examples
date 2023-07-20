import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取训练数据集
with open("train.txt", "r", encoding="utf-8") as file:
    texts = file.read().split('\n')
    texts = [item.strip() for item in texts if item.strip() != '']


max_length = 1024


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
        # input_attention_mask = input_tokenizer['attention_mask'][:-1]
        label = input_ids[1:]
        return {
            'input_index': input_index,
            # 'input_attention_mask': torch.tensor(input_attention_mask, dtype=torch.long),
            'label':  label
        }

    def __len__(self):
        return len(self.text)


class OptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        print(self.model)

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
    # input_attention_mask = torch.nn.utils.rnn.pad_sequence([item['input_attention_mask'] for item in batch_data],
    #                                                        batch_first=True, padding_value=0)
    label = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch_data],
                                            batch_first=True,
                                            padding_value=0)
    input_index = input_index.to(device)
    # input_attention_mask = input_attention_mask.to(device)
    label = label.to(device)
    return {
        'input_index': input_index,
        # 'input_attention_mask': input_attention_mask,
        'label': label
    }


train_dataset = MyDataset(texts)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=process)

model = OptModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
epoch = 6

for e in range(epoch):
    for index, data in enumerate(train_dataloader):
        data['input_index'] = data['input_index'].to(device)
        data['label'] = data['label'].to(device)

        optim.zero_grad()
        loss = model(data['input_index'], data['label'])
        loss.backward()
        optim.step()

        print(f'loss:{loss.item()}')


