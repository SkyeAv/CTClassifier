from __future__ import annotations
from CTC.export import histogram_density_plot
from CTC.export import predictions_with_shap
from CTC.export import precision_recall_plot
from CTC.export import relational_database
from CTC.training import train_model
from collections.abc import Mapping
from CTC.export import predictions
from CTC.export import shap_plot
from CTC.dataset import labelset
from CTC.export import roc_plot
from CTC.dataset import dataset
from CTC.dataset import prodset
from CTC.store import gen_hash
from CTC.store import STORE
import numpy.typing as npt
from CTC.yaml import load
from pathlib import Path
from typing import Any
import numpy as np
import typer

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
  database: Path = typer.Option(..., "-d", "--database", help="01 | Path to database.yaml"),
  booster: Path = typer.Option(..., "-b", "--booster", help="02 | Path to booster.yaml"),
) -> None:
  database_cfg: dict[str, Any] = load(database)
  _dataset, dataset_hash = dataset(database_cfg)

  booster_cfg: dict[str, Any] = load(booster)
  seed: int = booster_cfg["seed"]
  dtrain, dtest, scale_pos_weight, label_hash, feature_names = labelset(
    _dataset,
    dataset_hash,
    booster_cfg["labels"],
    booster_cfg["test_size"],
    seed
  )

  booster_hash: str = gen_hash(repr(booster_cfg) + dataset_hash + label_hash)
  model_store: Path = STORE / "MODELS" / booster_hash
  model_store.mkdir(parents=True, exist_ok=True)

  params: dict[str, Any] = booster_cfg["params"]
  assert isinstance(params, Mapping), "05 | Booster params must be a dict"
  params["seed"] = seed
  params["device_type"] = "cpu"
  params["scale_pos_weight"] = scale_pos_weight
  params["num_threads"] = -1

  booster: Path = train_model(dtrain, dtest, params, model_store)
  X, trials = prodset(_dataset)
  preds: npt.NDArray[np.float64] = predictions(booster, X)
  #relational_database(model_store, preds, trials)
  histogram_density_plot(model_store, preds)

  dtest.construct()
  X_test: npt.NDArray[np.float64] = dtest.get_data()
  preds_test, shap_values = predictions_with_shap(booster, X_test)
  shap_plot(shap_values, model_store, X_test, feature_names)

  #true_test: npt.NDArray[np.float64] = dtest.get_label()
  #w_test: npt.NDArray[np.float64] = dtest.get_weight()
  #precision_recall_plot(model_store, preds_test, true_test, w_test)
  #roc_plot(model_store, preds_test, true_test, w_test)
