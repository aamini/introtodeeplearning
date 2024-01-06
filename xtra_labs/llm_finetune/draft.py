"""
Drafting lab flow in script format using PyTorch
"""
from datasets import load_dataset
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from trl import SFTTrainer

from utils import run_benchmark, make_spider_plot

# Part 1

# TEXT: overview of LLM lab
# Load pretrained LLM (medium size model)

model_name = "facebook/opt-125m"
# model_name = "facebook/opt-1.3b"
# had to load non TF version to run benchmarking code
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# TEXT: explain tokenizer
# Include cell for tokenizer inspection

# TEXT: explain how LLMs are trained for next token prediction 
# Write a function to predict next token
def predict_next_token(probs, tokenizer):
    new_token = np.random.choice(len(probs), p=probs.numpy())
    print(tokenizer.decode(new_token), end='', flush=True)
    return new_token

# TEXT: explain that next token prediction must be called multiple times for inference
# Call in loop for autoregressive inference
def generate(start_text, model, tokenizer, num_steps=20, temp=1.): 
    print(start_text, end="")
    x = tokenizer.encode(start_text)
    num_start = len(x)

    for i in range(num_steps):
        input_tensor = torch.tensor(x).view(1, -1).to("cuda")
        logits = model(input_tensor).logits
        probs = F.softmax(logits/temp, dim=-1)[0, -1, :].cpu().detach()

        new_token = predict_next_token(probs, tokenizer)
        x.append(new_token)
    
    output = tokenizer.decode(x[num_start:])
    return output

# Test autoregressive generation
# while True: 
#     print("\n\n\n\n\n")
#     input_text = input("Prompt: ")
#     output = generate(input_text, model, tokenizer)

# TEXT: some background on LLM benchmarking
# Load benchmark dataset and evaluate model
dataset = pd.read_csv("benchmark.csv")
# category_accs_1300m, avg_acc_1300m = run_benchmark(model, tokenizer, dataset)

# TEXT: ask them to make a prediction on how accuracy will be affected by different model sizes

# Benchmark smaller model
# model_name_350m = "facebook/opt-350m" 
# model_350m = transformers.AutoModelForCausalLM.from_pretrained(model_name_350m, device_map="auto")
# tokenizer_350m = transformers.AutoTokenizer.from_pretrained(model_name_350m)

# category_accs_350m, avg_acc_350m = run_benchmark(model_350m, tokenizer_350m, dataset)

# Benchmark larger model
# model_name_2700m = "facebook/opt-2.7b" 
# model_2700m = transformers.AutoModelForCausalLM.from_pretrained(model_name_2700m, device_map="auto")
# tokenizer_2700m = transformers.AutoTokenizer.from_pretrained(model_name_2700m)

# category_accs_2700m, avg_acc_2700m = run_benchmark(model_2700m, tokenizer_2700m, dataset)

# Spider plot

# benchmark_data = {"350M-Model": category_accs_350m, "1300M-Model": category_accs_1300m, "2700M-Model": category_accs_2700m}
# make_spider_plot(benchmark_data)

# Part 2

# inspect current model
# print(model)

# new LoRA linear layer class
class LoRALinear(nn.Linear):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            pretrained_weight: torch.Tensor,
            r: int = 8,
            lora_alpha: int = 1,
            **kwargs
    ):
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.lora_alpha = lora_alpha

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.weight.data = pretrained_weight

        # from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            result = F.linear(x, self.weight, bias=self.bias)            
            result += (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

# replace linear layers in model recursively
def replace_linear_with_lora(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child.in_features, child.out_features, child.weight))
            break
        else:
            replace_linear_with_lora(child)

replace_linear_with_lora(model)

# inspect new model
# print(model)

# load chat dataset
dataset_name = "timdettmers/openassistant-guanaco"
ft_dataset = load_dataset(dataset_name, split="train")

# train model (barebones loop)
batch_size = 4
context_length = 768

model = model.to("cuda")
for batch in ft_dataset:
    prompt = batch["text"]
    
    # encode with tokenizer
    x = tokenizer.encode(prompt)
    x_tensor = torch.tensor(x).view(1, -1).to("cuda")
    input_tensor = x_tensor[:,:context_length]
    target_next_word = x_tensor[:,context_length]

    # run through model
    logits = model(input_tensor).logits

    probs = F.softmax(logits, dim=-1)[0, -1, :].cpu().detach()
    new_token = np.random.choice(len(probs), p=probs.numpy())
    print(tokenizer.decode(new_token), end='', flush=True)

    # apply loss


# evaluate finetuned model on benchmark
category_accs_1300m_ft, avg_acc_1300m_ft = run_benchmark(model, tokenizer, dataset)

# add to spider plot 
# benchmark_data = {"350M-Model": category_accs_350m, "1300M-Model": category_accs_1300m, "1300M-Model-Finetuned": category_accs_1300m_ft, "2700M-Model": category_accs_2700m}
# make_spider_plot(benchmark_data)