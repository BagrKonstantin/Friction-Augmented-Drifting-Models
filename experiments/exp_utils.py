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
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    timestamp = datetime.now().strftime("%d-%m-%Y")
    run_name = f"{timestamp}_{method_name}_{seed}"
    run_dir = os.path.join("experiments", "runs", run_name)
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
