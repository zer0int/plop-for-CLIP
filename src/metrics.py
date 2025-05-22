import torch
from collections import defaultdict
import numpy as np

def calculate_nfn_scores(model, batch, random_baseline=True):
    """
    Calculate NFN scores for all weight matrices.
    Args:
        model: Model to calculate NFN scores for.
        batch: Batch of problems.
        random_baseline: Whether to calculate the random baseline (this is True by default since it's needed for the NFN score).
    Returns:
        Dictionary of NFN scores for all weight matrices.
    """
    # Move batch to GPU if needed
    if next(model.parameters()).device != batch['input_ids'].device:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        
    # Initialize metrics dictionary
    metrics = defaultdict(dict)
    
    # Define hook function to calculate NFN scores for each weight matrix
    def hook_fn(name):
        """
        Hook function to calculate NFN scores for each weight matrix.
        Args:
            name: Name of the weight matrix.
        Returns:
            Hook function to calculate NFN scores for each weight matrix.
        """
        # Define inner hook function to calculate NFN scores for each weight matrix
        def hook(module, input, output):
            """
            Inner hook function to calculate NFN scores for each weight matrix.
            Args:
                module: Module to calculate NFN scores for.
                input: Input to the module.
                output: Output from the module (won't be used here).
            """
            if hasattr(module, 'weight') and module.weight is not None:
                # Get input and weight matrices
                z = input[0] if isinstance(input, tuple) else input
                W = module.weight
                z = z.float()
                W = W.float()
                
                # Reshape input if it's a 3D tensor
                if len(z.shape) > 2:
                    batch_size, seq_len, hidden_dim = z.shape
                    z = z.reshape(-1, hidden_dim)
                
                # Calculate NFN scores
                try:
                    # We calculate the Frobenius norm of W to normalize W for stability, but it is not necessary.
                    W_norm = (W**2).mean().sqrt()
                    z_norm = torch.norm(z, dim=1, keepdim=True)
                    W_normalized = W / (W_norm + 1e-8)
                    z_normalized = z / (z_norm + 1e-8)
                    Wz = torch.mm(z_normalized, W_normalized.t())
                    metrics[name]['actual'] = torch.norm(Wz, dim=1).mean().item()/z.shape[1]
                    if random_baseline:
                        z_random = torch.randn_like(z_normalized)
                        z_random_norm = torch.norm(z_random, dim=1, keepdim=True)
                        z_random_normalized = z_random / (z_random_norm + 1e-8)
                        Wz_random = torch.mm(z_random_normalized, W_normalized.t())
                        metrics[name]['random'] = torch.norm(Wz_random, dim=1).mean().item()/z.shape[1]
                except RuntimeError as e:
                    print(f"Error in layer {name}:")
                    print(f"Input shape: {z.shape}")
                    print(f"Weight shape: {W.shape}")
                    raise e
        return hook
    hooks = []
    for name, module in model.named_modules():
        embedding_filter = isinstance(module, torch.nn.Embedding)
        ln_filter = isinstance(module, torch.nn.LayerNorm) or 'norm' in name.lower()
        if hasattr(module, 'weight') and (module.weight is not None) and (not embedding_filter) and (not ln_filter):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        _ = model(**batch)
    for hook in hooks:
        hook.remove()

    return metrics

def get_group_metrics(metrics, groups=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'down_proj']):
    """
    Calculate group metrics.
    Args:
        metrics: Dictionary of NFN scores for all weight matrices.
        groups: List of groups to calculate metrics for.
    Returns:
        Dictionary of group metrics.
    """
    group_metrics = defaultdict(dict)
    for group in groups:
        group_metrics[group] = {
            'count': 0,
            'actual_sum': 0.0,
            'random_sum': 0.0
        }
    for name, values in metrics.items():
        for group in groups:
            if group in name:
                group_metrics[group]['count'] += 1
                group_metrics[group]['actual_sum'] += values.get('actual', 0.0)
                group_metrics[group]['random_sum'] += values.get('random', 0.0)
    results = {}
    for group, data in group_metrics.items():
        count = data['count']
        if count > 0:
            results[group] = {
                'actual': data['actual_sum'] / count,
                'random': data['random_sum'] / count if 'random_sum' in data else 0.0,
                'nfn': data['actual_sum'] / data['random_sum'] if 'random_sum' in data else 0.0
            }
        else:
            results[group] = {'actual': 0.0, 'random': 0.0, 'nfn': 0.0}
    return results 