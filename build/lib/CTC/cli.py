from __future__ import annotations
from CTC.export import relational_database
from CTC.training import train_model
from CTC.dataset import labelset
from CTC.dataset import dataset
from CTC.dataset import prodset
from CTC.store import gen_hash
from CTC.yaml import load
from pathlib import Path
from typing import Any
import typer

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
  database: Path = typer.Option(..., "-d", "--database", help="01 | Path to database.yaml"),
  booster: Path = typer.Option(..., "-b", "--booster", help="02 | Path to booster.yaml"),
  sqlite: Path = typer.Option("CTC.sqlite3", "-s", "--sqlite", help="03 | Path to save CTC sqlite")
) -> None:
  database_cfg: dict[str, Any] = load(database)
  _dataset, dataset_hash = dataset(**database_cfg)

  booster_cfg: dict[str, Any] = load(booster)
  seed: int = booster_cfg["seed"]
  dtrain, dtest, scale_pos_weight, label_hash = labelset(
    _dataset,
    booster_cfg["labels"],
    booster_cfg["test_size"],
    seed
  )

  booster_hash: str = gen_hash(repr(booster_cfg) + dataset_hash + label_hash)
  params: dict[str, Any] = booster_cfg["params"]
  assert isinstance(params, dict[str, Any]), "05 | Booster params must be a dict"
  params["seed"] = seed
  params["device_type"] = "cuda"

  booster: Path = train_model(dtrain, dtest, params, booster_hash)
  X, trials = prodset(_dataset)
  relational_database(booster, X, trials, sqlite)
