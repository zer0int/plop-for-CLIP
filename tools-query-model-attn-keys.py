import os
import attnclip as clip
import torch

device="cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-L/14", device=device, jit=False)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name, module)
