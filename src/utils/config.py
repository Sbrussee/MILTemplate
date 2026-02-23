from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    return cfg