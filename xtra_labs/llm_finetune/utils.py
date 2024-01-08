"""
Contains functions that the students will not interface with
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm

def run_benchmark(model, tokenizer, dataset, few_shot=7, num_steps=500, verbose=False):
    device = model.device
    dataset["Correct"] = 0.0

    # Loop through every question in the benchmark
    for step, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = row['Question']
        pre_text = f"### Human: {question}### Assistant:"
        len_prefix = len(tokenizer.encode(pre_text))
        
        # Run the model individually with each of the four responses. 
        # Measure the model's logprob for outputing each of the four responses. 
        # Choose the answer with the highest logprob
        logprobs = []
        answers = []
        for choice in ["A", "B", "C", "D"]: 
            answer = row[f'Answer {choice}']
            text = f"{pre_text} {answer}"

            # Run the model 
            with torch.no_grad():
                x = tokenizer.encode(text, return_tensors="pt").to(device)
                logits = model(x).logits
                probs = F.softmax(logits, dim=-1)[0, :-1, :]  # shape: [seq_len-1, vocab_size]
                y = x[0, 1:]  # shape: [seq_len-1]

            # Compute the log probability for this answer to appear (average logprob over the answer tokens)
            next_token_prob = np.array([probs[i, y[i]].item() for i in range(y.shape[0])])
            num_ans_tokens = x.shape[1] - len_prefix
            logprob = np.mean(np.log(next_token_prob[-num_ans_tokens:]))
            logprobs.append(logprob)
            answers.append(answer)
        
        # Check for the correct answer (always the zero-th index, by definition)
        correct = np.argmax(logprobs) == 0

        # Record if the model got the answer correct or not. 
        # Optionally print the question -> prediction if verbose
        dataset.at[step, "Correct"] = float(correct)
        if verbose: 
            print(f"[{correct}] {question} -> {answers[np.argmax(logprobs)]}")

    
    # Group by the the categories and compute the average accuracy
    accs = dataset.groupby("Category")["Correct"].mean()
    sorted_accs = accs.sort_values()
    print(sorted_accs)

    return accs, dataset["Correct"].mean()

def make_spider_plot(data):
    """
    Data is a dictionary where keys are different entities
    Values are pd Series where series indices are plot labels and series values show performance
    """
    colors = ['#1aaf6c', '#429bf4', '#d42cea']
    i = 0
    fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(polar=True))
    for k,v in data.items():
        labels = v.index.tolist()
        values = v.values.tolist()
        
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, color=colors[i], linewidth=1, label=k)
        ax.fill(angles, values, color=colors[i], alpha=0.25)

        i+=1

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(0, 1)
    ax.set_rlabel_position(180 / num_vars)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig("spider.png")


        