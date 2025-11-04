from __future__ import annotations
import numpy.typing as npt
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import numpy as np
import sqlite3

def relational_database(
  booster: Path,
  X: npt.NDArray[np.float16],
  trials: npt.NDArray[np.string_],
  save: Path
) -> None:
  bst: lgb.Booster = lgb.Booster(model_file=booster)
  _iter = bst.best_iteration
  pred_labels: npt.NDArray[np.float32] = bst.predict(X, _iter, raw_score=True).ravel().astype(np.float32)
  
  df: pd.DataFrame = pd.DataFrame({"NCT_ID": trials, "LABEL": pred_labels})
  df = df.set_index("NCT_ID")
  with sqlite3.connect(save.as_posix()) as conn:
    df.to_sql("TRUSTWORTHYNESS", conn, if_exists="replace", index=True)
