from __future__ import annotations
from yaml.loader import Loader
from pathlib import Path
from typing import Any
import yaml

def load(p: Path) -> dict[str, Any]:
  with p.open("rt") as f:
    raw: Any = yaml.load(f, Loader=Loader)
    assert isinstance(raw, dict[str, Any]), f"04 | Configuration {p} must not be a list"
    return raw
