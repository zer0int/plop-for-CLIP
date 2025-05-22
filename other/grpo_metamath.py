#!/usr/bin/env python3
"""
GRPO implementation, adapted from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_(1B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9
"""

import argparse
from datasets import load_dataset
import numpy as np
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from args import parse_args




# Set random seed for reproducibility
np.random.seed(42)

######### reasoning structure ##############
reasoning_start = "<think>"
reasoning_end   = "</think>"
solution_start = "<answer>"
solution_end = "</answer>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start} and {solution_end}"""

############# dataset ################

dataset = load_dataset("meta-math/MetaMathQA", split="train")
filtered_dataset = dataset.filter(lambda x: x["type"] in ["GSM_Rephrased", "GSM_SV", "GSM_AnsAug"] )
filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(50000))


def extract_hash_answer(text):
    if "The answer is: " not in text: return None
    return text.split("The answer is: ")[1].strip()


processed_dataset = filtered_dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["query"]},
    ],
    "answer": extract_hash_answer(x["response"]),
})


################ reward functions ###################

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
    )
    

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(solution_start)  == 1 else -0.5
        score += 0.5 if response.count(solution_end)    == 1 else -0.5
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess       = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores




def main():
    # Parse command line arguments
    args = parse_args()
    
    # init wandb
    wandb.init(project="GRPO_MetaMath")

    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=args.target_modules,
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    
    
    
    
    print(f"Configuring GRPO...")
    
    training_args = GRPOConfig(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    #num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=512,  # default: 256
    num_generations=args.num_generations,  # default: 8
    max_prompt_length=256,  # default: 512
    # Parameters related to reporting and saving,
    #log_completions=True,
    #num_completions_to_print=1,
    per_device_train_batch_size=args.per_device_train_batch_size,
    logging_steps=args.logging_steps,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=50,
    report_to=["wandb"],
    max_grad_norm=1.0
    )

    
    # Run GRPO
    print("\nRunning GRPO ...")
    trainer = GRPOTrainer(
        model=model, 
        processing_class=tokenizer, 
        reward_funcs=[match_format_exactly, match_format_approximately, check_answer, check_numbers], 
        args=training_args, 
        train_dataset=processed_dataset
        )
    
    trainer.train()
    

if __name__ == "__main__":
    main()