from __future__ import annotations
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torchdr import IncrementalPCA
from dataclasses import dataclass
import torch.nn.functional as F
from CTC.store import gen_hash
from functools import reduce
from CTC.store import STORE
from typing import Optional
import numpy.typing as npt
from pathlib import Path
from typing import Self
from torch import optim
import lightgbm as lgb
from typing import Any
from torch import nn
import polars as pl
import numpy as np
import torch
import lzma

def _load_table(p: Path, table: str) -> pl.DataFrame:
  with lzma.open(p, mode="rb") as f:
    df: pl.DataFrame = pl.read_csv(f, has_header=True, separator="|")
    df = df.drop("id", strict=False)
    return df.rename({col: f"{table}__{col}" for col in df.columns if col != "nct_id"})

def _inner_join(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
  return df1.join(df2, on="nct_id", how="inner")

def _read_snapshot(archive: Path, tables: list[str]) -> pl.DataFrame:
  dfs: list[pl.DataFrame] = []
  for table in tables:
    p: Path = archive / (table + ".txt.xz")
    df: pl.DataFrame = _load_table(p, table)
    dfs.append(df)

  return reduce(_inner_join, dfs)

def _remove_device_trials(
  df: pl.DataFrame,
  fda: str = "studies__is_fda_regulated_device",
  unapproved: str = "studies__is_unapproved_device"
) -> pl.DataFrame:
  _expr_FDA: pl.Expr = (pl.col(fda) != "t") | (pl.col(fda).is_null())
  _expr_unapproved: pl.Expr = (pl.col(unapproved) != "t") | (pl.col(unapproved).is_null())
  return df.filter(_expr_FDA & _expr_unapproved)

def _exclude(df: pl.DataFrame, exclude: list[str]) -> pl.DataFrame:
  return df.select(pl.exclude(exclude))

def _schema(df: pl.DataFrame) -> dict[str, Any]:
  schema: dict[str, Any] = dict(df.schema)
  del schema["nct_id"]
  return schema

def _cardinality(df: pl.DataFrame, string: list[str]) -> dict[str, int]:
  cardinality: pl.DataFrame = df.select(pl.col(string).n_unique())
  return cardinality.row(0, named=True)

def _col_lists(df: pl.DataFrame, threshold: int) -> tuple[list[str]]:
  schema: dict[str, Any] = _schema(df)
  string: list[str] = [col for col, dtype in schema.items() if dtype == pl.String]
  numeric: list[str] = [col for col in schema.keys() if col not in string]
  embeddings: list[str] = [col for col, n in _cardinality(df, string).items() if n > threshold]
  hot_one: list[str] = [col for col in string if col not in embeddings]
  return numeric, embeddings, hot_one

def _z_scores(df: pl.DataFrame, numeric: list[str]) -> pl.DataFrame:
  for col in numeric:
    _expr: pl.Expr = pl.col(col).cast(pl.Float64)
    df = df.with_columns(((_expr - _expr.mean()) / _expr.std()).alias(col))
  return df

def _hot_one_encode(df: pl.DataFrame, hot_one: list[str]) -> pl.DataFrame:
  return df.to_dummies(columns=hot_one, separator="__")

@torch.no_grad()
def _ipca(X: npt.NDArray[np.float32], device: str = "cuda:1", hidden: int = 16) -> tuple[list[slice], torch.Tensor]:
  samples, features = X.shape
  batch: int = (5 * features)
  ipca: IncrementalPCA = IncrementalPCA(n_components=hidden, device=device, lowrank=True)
  batch_iter: list[slice] = list(ipca.gen_batches(n=samples, batch_size=batch, min_batch_size=hidden))

  for sl in batch_iter:
    xb: torch.Tensor = torch.from_numpy(X[sl]).to(device, non_blocking=True)
    ipca.partial_fit(xb)
    del xb
    torch.cuda.synchronize(device)

  chunks: list[torch.Tensor] = []
  for sl in batch_iter:
    xb: torch.Tensor = torch.from_numpy(X[sl]).to(device, non_blocking=True)
    T = ipca.transform(xb)
    chunks.append(T)
    del xb
    del T
    torch.cuda.synchronize(device)

  out: torch.Tensor = torch.cat(chunks, dim=0)
  del chunks
  torch.cuda.synchronize(device)
  return batch_iter, out.contiguous()

class AutoEncoder(nn.Module):
  def __init__(
    self: Self,
    indim: int = 16,
    hidden1: int = 24,
    hidden2: int = 12,
    outdim: int = 8,
    dropout: float = 0.1
  ) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(indim, hidden1),
      nn.Dropout(dropout),
      nn.GELU(),
      nn.Linear(hidden1, hidden2),
      nn.GELU(),
      nn.Linear(hidden2, outdim)
    )
    self.decoder = nn.Sequential(
      nn.Linear(outdim, hidden2),
      nn.GELU(),
      nn.Linear(hidden2, hidden1),
      nn.GELU(),
      nn.Linear(hidden1, indim),
    )
    return None

  def encode(self: Self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)

  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x)
    x_hat = self.decoder(z) 
    return x_hat

def _train(
  model: AutoEncoder,
  optimizer: optim.AdamW,
  device: str,
  batch_iter: list[slice],
  T: torch.Tensor,
  norm: float = 1.0
) -> None:
  for sl in batch_iter:
    xb: torch.Tensor = T[sl]
    optimizer.zero_grad()
    xb_hat: torch.Tensor = model.forward(xb)
    loss: torch.Tensor = F.mse_loss(xb_hat, xb, reduction="mean")
    del xb
    del xb_hat

    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=norm)
    optimizer.step()
    torch.cuda.synchronize(device)

@torch.inference_mode()
def _test(
  model: AutoEncoder,
  device: str,
  batch_iter: list[slice],
  T: torch.Tensor
) -> float:
  total_loss: float = 0.0
  n: int = 0

  for sl in batch_iter:
    xb: torch.Tensor = T[sl]
    bs: int = xb.size(0)
    xb_hat: torch.Tensor = model.forward(xb)
    loss: torch.Tensor = F.mse_loss(xb_hat, xb, reduction="mean")

    del xb
    del xb_hat
    total_loss += loss.item() * bs
    n += bs
    torch.cuda.synchronize(device)

  return total_loss / n

@dataclass
class EarlyStopping:
  patience: int = 7
  min_delta: float = 1e-4
  best: float = np.inf
  wait: int = 0
  stopped_at: int = 0

  def step(
    self: Self,
    loss: float,
  ) -> bool:
    improved: bool = (self.best - loss) > self.min_delta

    if improved:
      self.best = loss
      self.wait = 0
    else:
      self.wait += 1

    return self.wait > self.patience

@torch.inference_mode()
def _encode(
  model: AutoEncoder,
  device: str,
  batch_iter: list[slice],
  T: torch.Tensor
) -> torch.Tensor:
  chunks: list[torch.Tensor] = []

  for sl in batch_iter:
    xb: torch.Tensor = T[sl]
    xb_hat: torch.Tensor = model.encode(xb)
    chunks.append(xb_hat)

    del xb
    del xb_hat
    torch.cuda.synchronize(device)

  out: torch.Tensor = torch.cat(chunks, dim=0)
  del chunks
  torch.cuda.synchronize(device)
  return out

def _autoencoder(
  batch_iter: list[slice],
  T: torch.Tensor,
  device: str = "cuda:1",
  epochs: int = 500,
  lr: float = 1e-4,
  decay: float = 0.0
) -> npt.NDArray[np.float64]:
  model: AutoEncoder = AutoEncoder().to(device)
  stopping = EarlyStopping()
  optimizer: optim.AdamW = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=decay
  )
  scheduler: optim.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-5,
    min_lr=1e-7
  )

  for _ in range(epochs):
    _train(model, optimizer, device, batch_iter, T)
    test: float = _test(model, device, batch_iter, T)
    scheduler.step(test)

    if stopping.step(test):
      break

  out: torch.Tensor = _encode(model, device, batch_iter, T)
  return out.detach().to("cpu").numpy().astype(np.float32)

def _reduce_noise(X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
  X = np.nan_to_num(X, 0.0, 0.0, 0.0)
  batch_iter, T = _ipca(X)
  return _autoencoder(batch_iter, T)

def _embed_texts(df: pl.DataFrame, embeddings: list[str], model: str, dataset_hash: str) -> pl.DataFrame:
  model: SentenceTransformer = SentenceTransformer(model)
  model.half()

  for col in embeddings:
    store: Path = STORE / "EMBEDDINGS" / (gen_hash(dataset_hash + col) + ".parquet")
    store.parent.mkdir(parents=True, exist_ok=True)

    if store.is_file():
      temp: pl.DataFrame = pl.read_parquet(store)
    else:
      texts: str = [str(x) if x else "" for x in df.get_column(col).to_numpy()]
      with torch.inference_mode():
        out: npt.NDArray[np.float64] = model.encode(
          texts,
          batch_size=4_096,
          device=[
            "cuda:0",
            "cuda:1",
            "cuda:2",
            "cuda:3"
          ],
          normalize_embeddings=True,
          chunk_size=16_384,
          convert_to_numpy=True,
          show_progress_bar=False
        ).astype(np.float32)

      out = _reduce_noise(out)
      shape: int = len(out[0])
      colnames: list[str] = [f"{col}__bert_{i}" for i in range(shape)]
      temp = pl.from_numpy(out, schema=colnames)
      temp.write_parquet(store)

    df = pl.concat((df.drop(col), temp), how="horizontal")

  return df

def dataset(cfg: dict[str, Any]) -> tuple[pl.DataFrame, str]:
  dataset_hash: str = gen_hash(cfg)
  store: Path = STORE / "DATASETS" / (dataset_hash + ".parquet")
  store.parent.mkdir(parents=True, exist_ok=True)

  if store.is_file():
    dataset: pl.DataFrame = pl.read_parquet(store)
  else:
    dataset = _read_snapshot(Path(cfg["archive"]), cfg["tables"])
    dataset = _remove_device_trials(dataset)

    exclude: Optional[list[str]] = cfg.get("exclude")
    if exclude:
      dataset = _exclude(dataset, exclude)

    numeric, embeddings, hot_one = _col_lists(dataset, cfg["threshold"])
    dataset = _z_scores(dataset, numeric)
    dataset = _hot_one_encode(dataset, hot_one)
    dataset = _embed_texts(dataset, embeddings, cfg["model"], dataset_hash)
    dataset.write_parquet(store)

  return dataset, dataset_hash

def _load_lables(p: Path) -> pl.DataFrame:
  return pl.read_csv(p, has_header=True, separator="\t")

def labelset(
  df: pl.DataFrame,
  dataset_hash: str,
  labels: Path,
  test_size: float,
  seed: int
) -> tuple[lgb.Dataset, float, str, npt.NDArray[np.string_]]:
  lf: pl.DataFrame = _load_lables(labels)
  label_hash: str = gen_hash(str(lf.to_init_repr()) + dataset_hash)
  labelset: pl.DataFrame = _inner_join(df, lf)

  exclude: list[str] = ["nct_id", "label", "weight"]
  X_frame: pl.DataFrame = labelset.select(pl.exclude(exclude))
  feature_names: npt.NDArray[np.string_] = np.array(df.columns) 

  X: npt.NDArray[np.float64] = X_frame.to_numpy().astype(np.float64)
  y: npt.NDArray[np.float64] = labelset.select("label").to_numpy().astype(np.float64).ravel()
  w: npt.NDArray[np.float64] = labelset.select("weight").to_numpy().astype(np.float64).ravel()

  ybar: float = np.mean(y)
  scale_pos_weight: float = ((1 - ybar) / ybar)

  X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, random_state=seed, stratify=y)
  dtrain: lgb.Dataset = lgb.Dataset(X_train, label=y_train, weight=w_train)
  dtest: lgb.Dataset = lgb.Dataset(X_test, label=y_test, weight=w_test, reference=dtrain)
  return dtrain, dtest, scale_pos_weight, label_hash, feature_names

def prodset(df: pl.DataFrame) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.string_]]:
  exclude: list[str] = ["nct_id"]
  X: npt.NDArray[np.float64] = df.select(pl.exclude(exclude)).to_numpy().astype(np.float64)
  trials: npt.NDArray[np.string_] = df.select("nct_id").to_numpy().astype(np.string_).ravel()
  return X, trials
