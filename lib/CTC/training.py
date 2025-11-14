from __future__ import annotations
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import lightgbm as lgb

def train_model(
  dtrain: lgb.Dataset,
  dtest: lgb.Dataset,
  params: dict[str, Any],
  model_store: Path
) -> Path:
  booster: Path = model_store / "booster.txt"
  if not booster.is_file():
    log: Path = model_store / "training.log"
    with log.open("w") as f:
      with redirect_stdout(f):

        bst: lgb.Booster = lgb.train(
          params=params,
          train_set=dtrain,
          num_boost_round=5_000,
          valid_sets=[dtest, dtrain],
          valid_names=["test", "train"],
          callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(10)
          ]
        )
        bst.save_model(booster)

  return booster
