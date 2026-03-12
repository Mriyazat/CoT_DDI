import os
import json
import random
import logging
import yaml
import torch
import numpy as np
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
            Path("configs/config.yaml"),
        ]
        for p in candidates:
            if p.exists():
                config_path = str(p)
                break
        if config_path is None:
            raise FileNotFoundError("Cannot find configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "No GPU available"
    lines = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        lines.append(f"  GPU {i}: {name} ({mem:.1f} GB)")
    return "\n".join(lines)


def ensure_dirs(cfg: dict):
    for d in [os.path.join(cfg["project"]["output_dir"], "figures"),
              os.path.join(cfg["project"]["output_dir"], "teacher_traces"),
              os.path.join(cfg["project"]["output_dir"], "checkpoints"),
              os.path.join(cfg["project"]["output_dir"], "results"),
              os.path.join(cfg["project"]["output_dir"], "logs")]:
        os.makedirs(d, exist_ok=True)


LABEL_CATEGORY_GROUPS = {
    "adverse_effects": "Risk/severity of adverse effects",
    "serum_increase": "Serum concentration increase",
    "serum_decrease": "Serum concentration decrease",
    "metabolism": "Metabolism changes",
    "activity_increase": "Therapeutic efficacy increase",
    "activity_decrease": "Therapeutic efficacy decrease",
    "absorption": "Absorption changes",
    "excretion": "Excretion rate changes",
    "other": "Other interactions",
}


def categorize_interaction(text: str) -> str:
    t = text.lower()
    if "risk or severity of adverse" in t:
        return "adverse_effects"
    if "serum concentration" in t and "increase" in t:
        return "serum_increase"
    if "serum concentration" in t and "decrease" in t:
        return "serum_decrease"
    if "metabolism" in t and "increase" in t:
        return "activity_increase"
    if "metabolism" in t and "decrease" in t:
        return "metabolism"
    if "metabolism" in t:
        return "metabolism"
    if "therapeutic efficacy" in t and "increase" in t:
        return "activity_increase"
    if "therapeutic efficacy" in t and "decrease" in t:
        return "activity_decrease"
    if "absorption" in t:
        return "absorption"
    if "excretion" in t:
        return "excretion"
    return "other"
