from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    """Load YAML configuration as OmegaConf ``DictConfig``."""
    return OmegaConf.load(path)
