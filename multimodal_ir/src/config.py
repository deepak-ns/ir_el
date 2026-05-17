# src/config.py
"""Central config loader. Import cfg from anywhere in the project."""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_ns(d):
    """Recursively convert dict to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str = "configs/config.yaml") -> SimpleNamespace:
    config_path = Path(path)
    if not config_path.exists():
        # Try relative to project root
        config_path = Path(__file__).parent.parent / path
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


cfg = load_config()
