from __future__ import annotations
from contextlib import redirect_stdout
from CTC.store import STORE
from pathlib import Path
from typing import Any
import lightgbm as lgb

def train_model(
  dtrain: lgb.Dataset,
  dtest: lgb.Dataset,
  params: dict[str, Any],
  booster_hash: str
) -> Path:
  store: Path = STORE / "MODELS" / booster_hash
  store.mkdir(parents=True, exist_ok=True)

  booster: Path = store / "booster.txt"
  if not booster.is_file():

    log: Path = store / "training.log"
    with log.open("w") as f:
      with redirect_stdout(f):
        bst: lgb.Booster = lgb.train(
          params=params,
          dtrain=dtrain,
          num_boost_round=5_000,
          valid_sets=[dtrain, dtest],
          valid_names=["train", "test"],
          callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(10)
          ]
        )
        bst.save_model(booster)

  return booster
