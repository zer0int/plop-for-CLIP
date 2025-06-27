import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from attnclip import load  # your patched CLIP
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import random
from cliptools import fix_random_seed

fix_random_seed() # deterministic backends, fixed seed

os.makedirs("results_attention_heads_interv", exist_ok=True)

# ---- DATASET (as in your code) ----

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
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = self.annotations[self.image_paths[idx]]
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]
        else:
            label = ''
        # We'll skip text for this script (pure image path)
        return image


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

# ---- LOAD MODEL ----

device = "cuda"
model, preprocess = load("ViT-L/14", device=device, jit=False)
model.eval().float()
model = model.to(device)

# ---------- HOOKS
# Register Neurons. For information about them + how to find them, see: https://github.com/zer0int/CLIP-test-time-registers
top_activations_layer_11 = [9, 987, 1967, 2555, 3661, 3784] 
top_activations_layer_12 = [42, 983, 1571, 2687, 3002, 3008, 3868]  

hooks_layer_11 = []
for feature_idx in top_activations_layer_11:
    hook = FeatureScalerHook(model, layer_idx=11, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
    hooks_layer_11.append(hook)

hooks_layer_12 = []
for feature_idx in top_activations_layer_12:
    hook = FeatureScalerHook(model, layer_idx=12, feature_idx=feature_idx, scale_factor=0, transformer_type='visual')
    hooks_layer_12.append(hook)
# ----------------

# ---- BATCH ----
BATCH_SIZE = 4
dataset = ImageTextDataset(
    "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square",
    "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-val-10_11.json",
    transform=preprocess
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
images = next(iter(loader))
images = images.cuda()

# ---- HOOK: Collect per-head L2s ----
vision_transformer = model.visual.transformer

num_layers = len(vision_transformer.resblocks)
num_heads = vision_transformer.resblocks[0].attn.num_heads

per_head_data = {l: [] for l in range(num_layers)}

def make_hook(layer_idx):
    def hook(module, input, output):
        seq_len, batch_size, embed_dim = output.shape
        num_heads = module.attn.num_heads
        head_dim = embed_dim // num_heads
        x = output.permute(1, 0, 2).contiguous()
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        head_norms = x.norm(dim=-1).mean(dim=(0, 1)).detach().cpu().numpy()
        per_head_data[layer_idx].append(head_norms)
    return hook

hooks = []
for l, block in enumerate(vision_transformer.resblocks):
    hooks.append(block.register_forward_hook(make_hook(l)))

with torch.no_grad():
    _ = model.visual(images)

for h in hooks:
    h.remove()

# ---- Aggregate results ----
per_head_mean = np.zeros((num_layers, num_heads))
for l in range(num_layers):
    arrs = per_head_data[l]
    arrs = np.stack(arrs, axis=0)
    per_head_mean[l] = arrs.mean(axis=0)

# ---- Save to CSV ----
csv_path = "results_attention_heads_interv/per_head_norms.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "head", "mean_l2"])
    for l in range(num_layers):
        for h in range(num_heads):
            writer.writerow([l, h, per_head_mean[l, h]])
print(f"Wrote per-head mean L2 norms to {csv_path}")

# ---- Summary Print ----
print("\nPer-head mean L2 norm (layer major, then head):")
for l in range(num_layers):
    vals = " ".join([f"{per_head_mean[l, h]:.3f}" for h in range(num_heads)])
    print(f"Layer {l:2d}: {vals}")

# ---- Cumulative sum per head ----
cumsum_per_head = per_head_mean.cumsum(axis=0)
final_cumsum = cumsum_per_head[-1]

print("\nCumulative sum of mean L2 norms (by head):")
for h in range(num_heads):
    print(f"Head {h:2d}: {final_cumsum[h]:.3f}")

# ---- Save cumulative sum CSV ----
cum_csv_path = "results_attention_heads_interv/per_head_cumsum.csv"
with open(cum_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["head", "cumsum"])
    for h in range(num_heads):
        writer.writerow([h, final_cumsum[h]])
print(f"Wrote cumulative sum per head to {cum_csv_path}")

# ---- Plot heatmap ----
plt.figure(figsize=(18, 6))
plt.imshow(per_head_mean.T, aspect="auto", cmap="viridis")
plt.colorbar(label="Mean L2 norm")
plt.xlabel("Block (Layer)")
plt.ylabel("Head")
plt.title("Vision Transformer: Per-head mean L2 norm (output tokens)")
plt.xticks(np.arange(num_layers))
plt.yticks(np.arange(num_heads))
plt.tight_layout()
plt.savefig("results_attention_heads_interv/per_head_l2norm_heatmap.png")
plt.close()
print("Saved per-head heatmap.")

# ---- Lineplot per head ----
plt.figure(figsize=(18, 5))
for h in range(num_heads):
    plt.plot(range(num_layers), per_head_mean[:, h], label=f"Head {h}")
plt.xlabel("Block (Layer)")
plt.ylabel("Mean L2 norm")
plt.title("Vision Transformer: Per-head mean L2 norm per block")
plt.legend(loc='upper right', ncol=4)
plt.tight_layout()
plt.savefig("results_attention_heads_interv/per_head_l2norm_lineplot.png")
plt.close()
print("Saved per-head lineplot.")

# ---- Plot cumulative sum per head ----
plt.figure(figsize=(12, 5))
plt.bar(range(num_heads), final_cumsum)
plt.xlabel("Head")
plt.ylabel("Cumulative sum of mean L2 norm")
plt.title("Vision Transformer: Cumulative sum per head (all blocks)")
plt.xticks(range(num_heads))
plt.tight_layout()
plt.savefig("results_attention_heads_interv/per_head_l2norm_cumsum.png")
plt.close()
print("Saved cumulative sum per head plot.")
