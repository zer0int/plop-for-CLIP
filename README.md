# PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models

This project provides a simple script to compute alignment metrics for transformer models on various datasets.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the main script:
```bash
python main.py --model <huggingface-model-handle> --dataset <math|code|history|logic> --batchsize <BATCHSIZE> --nbsamples <N> --seqlen <SEQ_LEN> --aggregation <type|layer|None> --output_dir <RESULTS_DIR>
```

Example:
```bash
python main.py --model meta-llama/Llama-3.2-1B-Instruct --dataset math --batchsize 8 --nbsamples 100 --seqlen 256 --aggregation type --output_dir results/
```

## Arguments
- `--model`: HuggingFace model handle (e.g., `google/gemma-2b`)
- `--dataset`: Dataset name (`math`, `code`, `history`, `logic`)
- `--batchsize`: Batch size (not used in this simple version, all samples are processed at once)
- `--nbsamples`: Number of samples to use from the dataset
- `--seqlen`: Sequence length for tokenization
- `--aggregation`: How to aggregate results (`type`, `layer`, or `None`)
- `--output_dir`: Directory to save results

## Output
- Raw and aggregated metrics are saved as JSON files in the specified output directory. 