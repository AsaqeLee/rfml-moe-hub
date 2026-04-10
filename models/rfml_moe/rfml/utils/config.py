"""Configuration loader with YAML support and nested attribute access."""

import yaml
from pathlib import Path
from typing import Any, Optional

_global_config = None


class Config(dict):
    """Dict subclass with attribute-style access and recursive nesting."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, Config):
                val = Config(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __repr__(self):
        return f"Config({dict.__repr__(self)})"

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        keys = dotted_key.split(".")
        val = self
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


def load_config(path: str | Path, overrides: Optional[dict] = None) -> Config:
    """Load YAML config file and apply optional overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config(raw)

    if overrides:
        _deep_update(cfg, overrides)

    global _global_config
    _global_config = cfg
    return cfg


def get_config() -> Config:
    """Return the globally loaded config."""
    if _global_config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _global_config


def _deep_update(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
