import torch
import numpy as np
import random
import os
import sys

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_default_device(device)
    return device

def setup_paths():
    # Set paths for ALAE and OFM imports
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    
    ofm_path = os.path.join(root_dir, "OFM")
    ofm_alae_path = os.path.join(root_dir, "OFM", "ALAE")
    
    if ofm_path not in sys.path:
        sys.path.append(ofm_path)
    if ofm_alae_path not in sys.path:
        sys.path.append(ofm_alae_path)
    
    return root_dir

def get_run_dir(method_name, seed):
    from datetime import datetime
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
    run_name = f"{timestamp}_{method_name}_{seed}"
    run_dir = os.path.join("scripts", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config(config, run_dir):
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def save_metrics(metrics, run_dir):
    import json
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def load_or_compute_target_stats(output_path, evaluator, Y_test, target_data):
    """
    Load or compute FID & CMMD statistics for the target distribution.
    
    Args:
        output_path: Directory where stats will be saved/loaded
        evaluator: MetricsEvaluator instance
        Y_test: Target test data
        target_data: Name of target dataset (for logging)
    
    Returns:
        tuple: (target_cmmd_act, target_fid_act, mu_target, sigma_target)
    """
    stats_file = os.path.join(output_path, "cmmd_fid.npz")
    
    if os.path.exists(stats_file):
        print("Loading precomputed FID & CMMD stats for target distribution...")
        stats = np.load(stats_file)
        target_cmmd_act = stats["target_cmmd_act"]
        target_fid_act = stats["target_fid_act"]
        mu_target = stats["mu_target"]
        sigma_target = stats["sigma_target"]
        print(f"Loaded target stats computed on {len(target_cmmd_act)} samples.")
    else:
        print(f"Precomputing FID & CMMD stats for {target_data} on the entire test set...")
        target_fid_act, target_cmmd_act = evaluator.get_eval_features(Y_test)
        mu_target, sigma_target = evaluator.get_stats(target_fid_act)
        print(f"Target stats computed on {len(target_fid_act)} samples.")
        np.savez(stats_file, 
                 target_cmmd_act=target_cmmd_act, 
                 target_fid_act=target_fid_act, 
                 mu_target=mu_target, 
                 sigma_target=sigma_target)
    
    return target_cmmd_act, target_fid_act, mu_target, sigma_target
