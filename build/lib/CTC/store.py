from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib

STORE: Path = Path("store")
STORE.mkdir(parents=True, exist_ok=True)

def gen_hash(x: Any) -> str:
  s: str = str(x).encode("utf-8")
  return hashlib.md5(s).hexdigest()
