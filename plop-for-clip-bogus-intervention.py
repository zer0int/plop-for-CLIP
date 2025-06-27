import os
import csv
import json
import random
import pprint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from metrics_clip import calculate_nfn_scores, get_group_metrics
import attnclip as clip
from PIL import Image
from cliptools import fix_random_seed

fix_random_seed() # deterministic backends, fixed seed


# Ensure output dirs exist
os.makedirs("results_csv_interv_bogus", exist_ok=True)
os.makedirs("results_plots_interv_bogus", exist_ok=True)

# --- Load model ---
model, preprocess = clip.load("ViT-L/14", device='cuda', jit=False)
model.eval().float()
model = model.cuda()



# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_idx, scale_factor, transformer_type='visual'):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.transformer_type = transformer_type
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        if self.transformer_type == 'visual':
            layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        else:
            layer = self.model.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension


train_dataset = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-train-0_9.json", transform=preprocess)
val_dataset = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-val-10_11.json", transform=preprocess)

# ---------- HOOKS
top_activations_layer_11 = [8, 986, 1966, 2554, 3660, 3783]
top_activations_layer_12 = [41, 982, 1570, 2686, 3001, 3007, 3867]

hooks_layer_11 = []
for feature_idx in top_activations_layer_11:
    hook = FeatureScalerHook(model, layer_idx=11, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
    hooks_layer_11.append(hook)

hooks_layer_12 = []
for feature_idx in top_activations_layer_12:
    hook = FeatureScalerHook(model, layer_idx=12, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
    hooks_layer_12.append(hook)
# ----------------

BATCH_SIZE = 8
loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
images, texts = next(iter(loader))
batch = {'image': images, 'text': texts}
batch = {k: v.cuda() for k, v in batch.items()}

# --- NFN calculation ---
metrics = calculate_nfn_scores(model, batch, random_baseline=True)

# --- Groupings: Vision + Text, by block and type ---
def parse_block_and_type(name):
    """
    Extracts block index and module type from a module name.
    Returns tuple: (branch, block, type), e.g. ('visual', 0, 'q_proj')
    """
    # visual.transformer.resblocks.11.attn.q_proj
    # transformer.resblocks.3.attn.q_proj
    parts = name.split('.')
    if parts[0] == 'visual':
        branch = "vision"
        block = int(parts[3])
    elif parts[0] == 'transformer':
        branch = "text"
        block = int(parts[2])
    else:
        return None
    type_candidate = parts[-1]
    if type_candidate in {'q_proj', 'k_proj', 'v_proj', 'out_proj', 'c_fc', 'c_proj'}:
        return branch, block, type_candidate
    return None

# 1. Layer-by-layer CSV export (all projections, all blocks)
per_block = {}
for name, values in metrics.items():
    parsed = parse_block_and_type(name)
    if parsed is None:
        continue
    branch, block, type_ = parsed
    key = (branch, block, type_)
    per_block[key] = values

csv_path = os.path.join("results_csv_interv_bogus", "plop_per_block.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["branch", "block", "type", "actual", "random", "nfn"])
    for (branch, block, type_), vals in sorted(per_block.items()):
        writer.writerow([branch, block, type_, vals.get("actual", 0.0), vals.get("random", 0.0), vals.get("nfn", 0.0)])

print(f"Wrote layer-by-layer results to {csv_path}")

# 2. Group summaries (across all blocks, per type and branch)
vision_types = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'c_fc', 'c_proj']
text_types = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'c_fc', 'c_proj']
summary = {}

for branch, groups in [("vision", vision_types), ("text", text_types)]:
    if branch == "vision":
        prefix = "visual.transformer.resblocks"
    else:
        prefix = "transformer.resblocks"
    metrics_branch = {k: v for k, v in metrics.items() if k.startswith(prefix)}
    group_result = get_group_metrics(metrics_branch, groups=groups, individual=False)
    summary[branch] = group_result
    # CSV export
    csv_branch = os.path.join("results_csv_interv_bogus", f"plop_summary_{branch}.csv")
    with open(csv_branch, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "actual", "random", "nfn"])
        for type_, vals in group_result.items():
            writer.writerow([type_, vals.get("actual", 0.0), vals.get("random", 0.0), vals.get("nfn", 0.0)])
    print(f"Wrote {branch} summary to {csv_branch}")

# 3. Plot overview
def plot_group(group_result, branch, save_dir="results_plots_interv_bogus"):
    types = list(group_result.keys())
    nfns = [group_result[t]['nfn'] for t in types]
    plt.figure(figsize=(10, 5))
    plt.bar(types, nfns)
    plt.title(f"PLoP NFN by type ({branch})")
    plt.ylabel("NFN")
    plt.ylim(0, max(nfns) * 1.2 if nfns else 1.0)
    plt.savefig(os.path.join(save_dir, f"nfn_{branch}.png"))
    plt.close()

plot_group(summary['vision'], "vision")
plot_group(summary['text'], "text")
print("Saved PLoP NFN plots for vision and text transformer.")

# 4. Optionally, plot per-block for each branch and type (heatmap or lineplot)
import numpy as np

def plot_per_block(per_block, branch, save_dir="results_plots_interv_bogus"):
    # block index -> type -> nfn
    block_types = sorted(set(k[2] for k in per_block if k[0] == branch))
    block_nums = sorted(set(k[1] for k in per_block if k[0] == branch))
    data = np.zeros((len(block_types), len(block_nums)))
    for i, t in enumerate(block_types):
        for j, b in enumerate(block_nums):
            vals = per_block.get((branch, b, t), {})
            data[i, j] = vals.get("nfn", 0.0)
    plt.figure(figsize=(16, 6))
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.yticks(range(len(block_types)), block_types)
    plt.xticks(range(len(block_nums)), block_nums)
    plt.xlabel("Block")
    plt.ylabel("Type")
    plt.title(f"NFN by Block and Type ({branch})")
    plt.colorbar(im, label="NFN")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"nfn_block_type_{branch}.png"))
    plt.close()

plot_per_block(per_block, "vision")
plot_per_block(per_block, "text")
print("Saved per-block heatmaps for vision and text transformer.")

# Final: print summary to console
pprint.pprint(summary)
