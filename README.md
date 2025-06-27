# PLoP for CLIP

- Uses 'import clip' / [OpenAI/CLIP](https://github.com/openai/CLIP)
- Modified version, exposing attn -> `attnclip` folder

0. Download dataset from [huggingface.co/datasets/SPRIGHT-T2I/spright_coco](https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco) (I provided the labels as .json here), or insert your own as `train_dataset`, `val_dataset` in all `plop-for-clip` code.
1. Run all `plop-for-clip*`
2. Run all `compare-clip*`
3. Run all `clip-attention`
4. Check out the results!

![results-plop-for-clip](https://github.com/user-attachments/assets/c883e5b1-7bd6-4854-9547-fbf85708022d)

### Register Neurons
- For more information on what 'register neurons' are and how to find them, see [github.com/zer0int/CLIP-test-time-registers](https://github.com/zer0int/CLIP-test-time-registers)
- Register Neuron Intervention vs. Bogus Neuron Intervention:

![CLIP-plop-intervention](https://github.com/user-attachments/assets/ef33eae5-22e5-473d-9520-9399c68c54ff)

- Attention L2 for individual heads:

![CLIP-before-after-heads](https://github.com/user-attachments/assets/78efb4d0-3ee5-452d-bdb3-be944276cc1e)

------
ORIGINAL README.MD BELOW
------

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
