import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/mnt/lorne/transformer_examples/llm/trained_model")

model = AutoModelForCausalLM.from_pretrained("/mnt/lorne/transformer_examples/llm/trained_model").cuda()
model = model.eval()


parser = argparse.ArgumentParser(description="Predict Argparse")
parser.add_argument("--question", "-q", help="问题内容")
args = parser.parse_args()

if __name__ == '__main__':
    input_context = args.question
    input_ids = tokenizer.encode(input_context, return_tensors='pt')
    input_ids = input_ids.cuda()
    output = model.generate(input_ids, max_length=50)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)
