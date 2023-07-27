import argparse

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Predict Argparse")
parser.add_argument("--question", "-q", help="问题内容")
parser.add_argument("--length", "-l", help="输出问题长度", default=100)
parser.add_argument("--top_k", "-k", help="top_k", default=50)
parser.add_argument("--top_p", "-p", help="top_p", default=0.95)
parser.add_argument("--temperature", "-t", help="temperature", default=0.05)
parser.add_argument("--do_sample", "-s", help="do_sample", default=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b")


class OptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b")
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

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
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# Load the model
state_dict = torch.load('internlm_chat_7b_model.pth', map_location=device)

# Remove 'module.' prefix
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)
model.eval()


if __name__ == '__main__':
    input_context = args.question
    max_length = int(args.length)
    top_k = int(args.top_k)
    top_p = float(args.top_p)
    temperature = float(args.temperature)
    do_sample = bool(args.do_sample)

    input_ids = tokenizer.encode(input_context, return_tensors='pt').to(device)

    output = model.generate(input_ids, max_length=max_length, do_sample=do_sample, top_k=top_k, top_p=top_p,
                            temperature=temperature)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print(output_text)
