from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from src.model.abmil import ABMIL
from src.model.mil import MIL

MODEL_REGISTRY: dict[str, type[MIL]] = {
    "abmil": ABMIL,
}


def create_model(model_cfg: Mapping[str, Any], in_dim: int) -> MIL:
    """Build a MIL model from configuration.

    Args:
        model_cfg: Model section from configuration.
        in_dim: Feature dimension ``D`` inferred from bags or config.

    Returns:
        Instantiated ``MIL`` model.

    Notes:
        ``model_cfg.name`` can be one of registered short names (e.g. ``abmil``) or
        a full import path ``package.module:ClassName``.
    """

    name = str(model_cfg.get("name", "abmil")).strip()
    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    kwargs["in_dim"] = int(in_dim)

    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](**kwargs)

    if ":" not in name:
        raise ValueError(f"Unknown model '{name}'. Use one of {list(MODEL_REGISTRY)} or module:Class syntax.")

    module_name, class_name = name.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    model = klass(**kwargs)
    if not isinstance(model, MIL):
        raise TypeError(f"Loaded model {name} is not a MIL subclass")
    return model
