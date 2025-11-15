from __future__ import annotations
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy.typing as npt
from pathlib import Path
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
import shap

def _load_booster(p: Path) -> lgb.Booster:
  return lgb.Booster(model_file=p)

def _invoke_booster(
  bst: lgb.Booster,
  X: npt.NDArray[np.float64],
  params: dict[str, bool]
) -> npt.NDArray[np.float64]:
  _iter = bst.best_iteration
  return bst.predict(X, _iter, **params).astype(np.float64)

def predictions(
  booster: Path,
  X: npt.NDArray[np.float64],
  params: dict[str, bool] = {"raw_score": True}
) -> npt.NDArray[np.float64]:
  bst: lgb.Booster = _load_booster(booster)
  return _invoke_booster(bst, X, params).ravel()

def predictions_with_shap(
  booster: Path,
  X: npt.NDArray[np.float64],
  params: dict[str, bool] = {"raw_score": True, "pred_contrib": True}
) -> tuple[npt.NDArray[np.float64]]:
  bst: lgb.Booster = _load_booster(booster)
  raw: npt.NDArray[np.float64] = _invoke_booster(bst, X, params)
  preds: npt.NDArray[np.float64] = raw[:, -1]
  shap_values: npt.NDArray[np.float64] = raw[:, :-1]
  return preds, shap_values

def relational_database(
  model_store: Path,
  preds: npt.NDArray[np.float64],
  trials: npt.NDArray[np.bytes_]
) -> Path:
  save: Path = model_store / "trials.sqlite3"
  df: pd.DataFrame = pd.DataFrame({"NCT_ID": trials, "LABEL": preds})
  df = df.set_index("NCT_ID")
  with sqlite3.connect(save.as_posix()) as conn:
    df.to_sql("TRUSTWORTHYNESS", conn, if_exists="replace", index=True)

def histogram_density_plot(
  model_store: Path,
  preds: npt.NDArray[np.float64],
) -> None:
  save: Path = model_store / "histogram_density_plot.png"
  ax: sns.ax = sns.histplot(x=preds, bins=30, binrange=(0, 1), stat="density", element="step")
  ax.set_xlabel(r"LightGBM Scores")
  ax.set_ylabel(r"Density")
  plt.tight_layout()
  plt.savefig(save, dpi=300)
  plt.close()

def roc_plot(
  model_store: Path,
  preds: npt.NDArray[np.float64],
  true: npt.NDArray[np.float64],
  weight: npt.NDArray[np.float64]
) -> None:
  save: Path = model_store / "roc_plot.png"
  roc_auc = roc_auc_score(true, preds, sample_weight=weight)
  fpr, tpr, _ = roc_curve(true, preds, sample_weight=weight)
  ax: sns.ax = sns.lineplot(x=fpr, y=tpr, color="blue", label=f"Model (ROC AUC = {roc_auc:.3g})")
  ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="No Skill")
  ax.set_xlabel(r"False Positive Rate (FPR)")
  ax.set_ylabel(r"True Positive Rate (TPR)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(save, dpi=300)
  plt.close()

def precision_recall_plot(
  model_store: Path,
  preds: npt.NDArray[np.float64],
  true: npt.NDArray[np.float64],
  weight: npt.NDArray[np.float64]
) -> None:
  save: Path = model_store / "precision_recall_plot.png"
  ap: np.float64 = average_precision_score(true, preds, sample_weight=weight)
  prec, rec, _ = precision_recall_curve(true, preds, sample_weight=weight)
  ax: sns.ax = sns.lineplot(x=prec, y=rec, color="blue", label=f"Average = {ap:.3g}")
  ax.set_xlabel(r"Recall")
  ax.set_ylabel(r"Precision")
  plt.legend()
  plt.tight_layout()
  plt.savefig(save, dpi=300)
  plt.close()


def shap_plot(
  shap_values: npt.NDArray[np.float64],
  model_store: Path,
  X: npt.NDArray[np.float64],
  feature_names: npt.NDArray[np.bytes_]
) -> None:
  save: Path = model_store / "shap_plot.png"
  shap.plots.violin(
    shap_values,
    features=X,
    feature_names=feature_names,
    plot_type="layered_violin",
    max_display=20,
    show=False
  )
  plt.tight_layout()
  plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.25)
  plt.close()
