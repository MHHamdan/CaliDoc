"""
Reproducibility Utilities

Functions for ensuring reproducible experiments.
"""

import os
import random
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def save_experiment_config(
    config: Dict[str, Any],
    output_dir: str,
    save_git: bool = True,
    save_env: bool = True
):
    """
    Save experiment configuration for reproducibility.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        save_git: Whether to save git commit info
        save_env: Whether to save environment info
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Save git commit
    if save_git:
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            diff = subprocess.check_output(
                ['git', 'diff', '--stat'],
                stderr=subprocess.DEVNULL
            ).decode()

            with open(output_path / 'git_info.txt', 'w') as f:
                f.write(f"Commit: {commit}\n\n")
                f.write(f"Diff:\n{diff}")
        except Exception:
            pass

    # Save environment
    if save_env:
        try:
            subprocess.run(
                ['pip', 'freeze'],
                stdout=open(output_path / 'requirements_frozen.txt', 'w'),
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    # Save PyTorch info
    pytorch_info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        pytorch_info['gpu_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    with open(output_path / 'pytorch_info.json', 'w') as f:
        json.dump(pytorch_info, f, indent=2)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file."""
    with open(config_path) as f:
        return json.load(f)
