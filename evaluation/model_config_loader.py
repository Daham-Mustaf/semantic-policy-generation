"""
Shared loader for evaluator model configs.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


MODEL_CONFIG_PATH = Path("evaluation/openai-apis/custom_models.json")


def load_model_config(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load evaluator model config from custom_models.json.

    Behavior:
    - If model_id is not provided: return the first config entry.
    - If model_id is provided: return the matching entry.
    """
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if not isinstance(configs, list) or not configs:
        raise ValueError(f"Model config file is empty or invalid: {MODEL_CONFIG_PATH}")

    if model_id is None:
        return configs[0]

    for config in configs:
        if config.get("model_id") == model_id:
            return config

    available = [c.get("model_id", "<missing>") for c in configs]
    raise ValueError(
        f"model_id '{model_id}' not found in {MODEL_CONFIG_PATH}. "
        f"Available model_id values: {available}"
    )
