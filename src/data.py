import torch
import random
from datasets import load_dataset
import string

# Only keep the minimal set of functions needed for the simplified main


def load_gsm8k_problems(num_samples=200):
    """
    Load problems from the GSM8K dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    Remark: For now, we only keep the questions. Answers can be included in the samples as well.
    """
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    import requests, json
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    problems = [json.loads(line)['question'] for line in lines[:num_samples*2]]
    return random.sample(problems, min(num_samples, len(problems)))

def load_mmlu_problems(subject="logical_fallacies", num_samples=200):
    """
    Load problems from the MMLU dataset.
    Args:
        subject: Subject to load.
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    problems = []
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        problem = f"{question}\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        problems.append(problem)
    return random.sample(problems, min(num_samples, len(problems)))

def load_humaneval_problems(num_samples=200):
    """
    Load problems from the HumanEval dataset.
    Args:
        num_samples: Number of samples to load.
    Returns:
        List of problems.
    """
    dataset = load_dataset("openai_humaneval", split="test")
    problems = [f"# Write a Python function\n{item['prompt']}" for item in dataset]
    return random.sample(problems, min(num_samples, len(problems)))

def prepare_batch(problems, tokenizer, max_length=256):
    """
    Prepare a batch of problems for the model.
    Args:
        problems: List of problems.
        tokenizer: Tokenizer.
        max_length: Maximum sequence length.
    Returns:
        Dictionary of encoded problems.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded = tokenizer(
        problems,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded

def get_dataset(dataset_name="math", num_samples=10, tokenizer=None):
    """
    Meta function to load a dataset.
    Args:
        dataset_name: Name of the dataset.
        num_samples: Number of samples to load.
        tokenizer: Tokenizer.
    Returns:
        List of problems.
    """
    if dataset_name == "gsm8k":
        return load_gsm8k_problems(num_samples)
    elif dataset_name == "mmlu_logic":
        return load_mmlu_problems("logical_fallacies", num_samples)
    elif dataset_name == "mmlu_history":
        return load_mmlu_problems("high_school_european_history", num_samples)
    elif dataset_name == "code":
        return load_humaneval_problems(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 