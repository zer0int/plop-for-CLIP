import argparse
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataset, prepare_batch
from src.metrics import calculate_nfn_scores, get_group_metrics
import json
from pprint import pprint
from collections import defaultdict


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))


def print_stylish_results(results, title="Results"):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    for k, v in results.items():
        print(f"{k:>20}: {v['nfn']}")
    print("="*60 + "\n")


def average_metrics(metrics_list):
    # metrics_list: list of dicts {name: {'actual': x, 'random': y}}
    if not metrics_list:
        return {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    avg_metrics = {}
    for k in all_keys:
        actuals = [m[k]['actual'] for m in metrics_list if k in m]
        randoms = [m[k]['random'] for m in metrics_list if k in m]
        avg_metrics[k] = {
            'actual': sum(actuals) / len(actuals) if actuals else 0.0,
            'random': sum(randoms) / len(randoms) if randoms else 0.0,
        }
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Run alignment metrics on a model and dataset.")
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model handle')
    parser.add_argument('--dataset', type=str, required=True, choices=['math', 'code', 'history', 'logic'], help='Dataset name')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size')
    parser.add_argument('--nbsamples', type=int, default=100, help='Number of samples to use from dataset')
    parser.add_argument('--seqlen', type=int, default=256, help='Sequence length')
    parser.add_argument('--aggregation', type=str, default='type', choices=['type', 'layer', 'None'], help='Aggregation type')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map dataset names
    dataset_map = {
        'math': 'gsm8k',
        'code': 'code',
        'history': 'mmlu_history',
        'logic': 'mmlu_logic',
    }
    dataset_name = dataset_map[args.dataset]

    # Get dataset
    print(f"Loading dataset: {args.dataset} ({dataset_name})")
    problems = get_dataset(dataset_name, args.nbsamples, tokenizer)
    print(f"Sampled {len(problems)} problems from dataset")
    if len(problems) > args.nbsamples:
        problems = random.sample(problems, args.nbsamples)

    # Split into batches
    batches = [problems[i:i+args.batchsize] for i in range(0, len(problems), args.batchsize)]
    print(f"Processing {len(batches)} batches of size up to {args.batchsize}")

    # Calculate metrics for each batch and average
    all_metrics = []
    for i, batch_problems in tqdm(enumerate(batches), total=len(batches)):
        print(f"Tokenizing batch {i+1}/{len(batches)}...")
        batch = prepare_batch(batch_problems, tokenizer, max_length=args.seqlen)
        print(f"Calculating alignment metrics for batch {i+1}/{len(batches)}...")
        metrics = calculate_nfn_scores(model, batch)
        all_metrics.append(metrics)

    avg_metrics = average_metrics(all_metrics)
    metrics_path = os.path.join(args.output_dir, args.model.split('/')[-1] + '_' + args.dataset + '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"Saved averaged metrics to {metrics_path}")

    # Aggregation
    agg_results = None
    if args.aggregation == 'type':
        agg_results = get_group_metrics(avg_metrics)
        agg_path = os.path.join(args.output_dir, args.model.split('/')[-1] + '_' + args.dataset + '_aggregated_by_type.json')
        with open(agg_path, 'w') as f:
            json.dump(agg_results, f, indent=2)
        print_stylish_results(agg_results, title="Aggregated by Module Type")
        print(f"Saved aggregated results to {agg_path}")
    elif args.aggregation == 'layer':
        # Find all layer numbers
        layer_numbers = set()
        for name in avg_metrics.keys():
            if 'layers.' in name:
                try:
                    layer_num = int(name.split('layers.')[1].split('.')[0])
                    layer_numbers.add(layer_num)
                except (IndexError, ValueError):
                    continue
        agg_results = get_group_metrics(avg_metrics, groups=[f'layers.{i}' for i in sorted(layer_numbers)])
        agg_path = os.path.join(args.output_dir, args.model.split('/')[-1] + '_' + args.dataset + '_aggregated_by_layer.json')
        with open(agg_path, 'w') as f:
            json.dump(agg_results, f, indent=2)
        print_stylish_results(agg_results, title="Aggregated by Layer")
        print(f"Saved aggregated results to {agg_path}")
    else:
        print_stylish_results(avg_metrics, title="Raw Metrics (No Aggregation)")

if __name__ == "__main__":
    main() 