"""
Drafting lab flow in script format using PyTorch
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from utils import run_benchmark, make_spider_plot

# Part 1

# TEXT: overview of LLM lab
# Load pretrained LLM (medium size model)

model_name = "facebook/opt-1.3b"
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
        input_tensor = tf.reshape(tf.constant(x), [1, -1])
        logits = model(input_tensor).logits
        probs = tf.nn.softmax(logits/temp)[0, -1, :]

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
category_accs_1300m, avg_acc_1300m = run_benchmark(model, tokenizer, dataset)

# TEXT: ask them to make a prediction on how accuracy will be affected by different model sizes

# Benchmark smaller model
model_name_350m = "facebook/opt-350m" 
model_350m = transformers.AutoModelForCausalLM.from_pretrained(model_name_350m, device_map="auto")
tokenizer_350m = transformers.AutoTokenizer.from_pretrained(model_350m)

category_accs_350m, avg_acc_350m = run_benchmark(model_350m, tokenizer_350m, dataset)

# Benchmark larger model
model_name_2700m = "facebook/opt-2.7b" 
model_2700m = transformers.AutoModelForCausalLM.from_pretrained(model_name_2700m, device_map="auto")
tokenizer_2700m = transformers.AutoTokenizer.from_pretrained(model_2700m)

category_accs_2700m, avg_acc_2700m = run_benchmark(model_2700m, tokenizer_2700m, dataset)

# Spider plot

benchmark_data = {"350M-Model": category_accs_350m, "1300M-Model": category_accs_1300m, "2700M-Model": category_accs_2700m}
make_spider_plot(benchmark_data)

# Part 2

# new LoRA linear layer class

# new attention layer class

# replace attention modules with new module

# load chat dataset

# train model

# evaluate finetuned model on benchmark

# add to spider plot 