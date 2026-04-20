from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List

import numpy as np
import pandas as pd


def load_positions_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ch_name", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in positions CSV: {missing}")
    return df.copy()


def pairwise_distances(df: pd.DataFrame) -> np.ndarray:
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def build_laplacian_knn(df: pd.DataFrame, k: int = 4) -> Dict[str, List[str]]:
    names = df["ch_name"].tolist()
    d = pairwise_distances(df)
    out = {}
    for i, name in enumerate(names):
        idx = np.argsort(d[i])[:k]
        out[name] = [names[j] for j in idx]
    return out


def build_bipolar_nearest(df: pd.DataFrame) -> Dict[str, str]:
    names = df["ch_name"].tolist()
    d = pairwise_distances(df)
    out = {}
    for i, name in enumerate(names):
        j = int(np.argmin(d[i]))
        out[name] = names[j]
    return out


def load_graph_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
