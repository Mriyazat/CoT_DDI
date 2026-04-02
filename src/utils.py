import os
import json
import random
import logging
import yaml
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


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
        cfg = yaml.safe_load(f)

    # On Compute Canada, redirect all large outputs to $SCRATCH
    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        cfg["project"]["output_dir"] = os.path.join(
            scratch, "ddi_v3_outputs"
        )
    return cfg


def setup_logging(name: str, log_dir: str = None) -> logging.Logger:
    if log_dir is None:
        scratch = os.environ.get("SCRATCH", "")
        if scratch:
            log_dir = os.path.join(scratch, "ddi_v3_outputs", "logs")
        else:
            log_dir = "outputs/logs"
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
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def gpu_info() -> str:
    if torch is None or not torch.cuda.is_available():
        return "No GPU available"
    lines = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        lines.append(f"  GPU {i}: {name} ({mem:.1f} GB)")
    return "\n".join(lines)


def ensure_dirs(cfg: dict):
    for d in [
        os.path.join(cfg["project"]["output_dir"], "figures"),
        os.path.join(cfg["project"]["output_dir"], "teacher_traces"),
        os.path.join(cfg["project"]["output_dir"], "checkpoints"),
        os.path.join(cfg["project"]["output_dir"], "results"),
        os.path.join(cfg["project"]["output_dir"], "logs"),
    ]:
        os.makedirs(d, exist_ok=True)


COARSE_CATEGORY_KEYWORDS = {
    "metabolism_decrease": ["metabolism of #Drug2 can be decreased", "metabolism of #Drug1 can be decreased"],
    "metabolism_increase": ["metabolism of #Drug2 can be increased", "metabolism of #Drug1 can be increased"],
    "serum_increase": ["serum concentration of #Drug2 can be increased", "serum concentration of #Drug1 can be increased",
                       "serum concentration of the active metabolite"],
    "serum_decrease": ["serum concentration of #Drug2 can be decreased", "serum concentration of #Drug1 can be decreased"],
    "adverse_effects": ["risk or severity of adverse effects"],
    "efficacy_decrease": ["therapeutic efficacy of #Drug2 can be decreased", "therapeutic efficacy of #Drug1 can be decreased"],
    "efficacy_increase": ["therapeutic efficacy of #Drug2 can be increased", "therapeutic efficacy of #Drug1 can be increased"],
    "excretion_decrease": ["may decrease the excretion rate", "excretion of #Drug2 can be decreased",
                           "excretion of #Drug1 can be decreased"],
    "excretion_increase": ["may increase the excretion rate", "excretion of #Drug2 can be increased",
                           "excretion of #Drug1 can be increased"],
    "absorption_decrease": ["absorption of #Drug2 can be decreased", "absorption of #Drug1 can be decreased"],
    "absorption_increase": ["absorption of #Drug2 can be increased", "absorption of #Drug1 can be increased"],
    "qtc_cardiac": ["qtc-prolonging", "qtc interval", "cardiac", "arrhythmia", "torsade"],
    "cns_effects": ["cns depressant", "serotonergic", "sedation", "respiratory depression"],
    "bleeding": ["hemorrhagic", "bleeding", "anticoagulant"],
    "nephrotoxicity": ["nephrotoxic", "renal"],
    "hepatotoxicity": ["hepatotoxic", "liver"],
    "hypotension": ["hypotensive"],
    "hyperkalemia": ["hyperkalemic", "hyperkalemia"],
}


def categorize_interaction(template: str) -> str:
    t = template.lower().replace("#drug1", "").replace("#drug2", "")
    for category, keywords in COARSE_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower().replace("#drug1", "").replace("#drug2", "") in t:
                return category
    return "other"
