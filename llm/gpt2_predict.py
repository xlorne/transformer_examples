import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


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

    def generate(self, input_ids, max_length=20, do_sample=True, top_k=50, top_p=0.95, temperature=0.05):
        output = self.model.generate(input_ids, max_length=max_length,
                            do_sample=True, top_k=50, top_p=0.95, temperature=0.05)
        return output


# Load the model
model = OptModel().to(device)
model.load_state_dict(torch.load('gpt2_model.pth'))
model.eval()


input_context = "诸葛亮与周瑜在赤壁之战交流了什么?"

input_ids = tokenizer.encode(input_context, return_tensors='pt').to(device)

output = model.generate(input_ids, max_length=50)

output_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(output_text)
