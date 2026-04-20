
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class CachedSubject:
    dataset_id: str
    subject: int
    channel_names: list[str]
    sfreq: float
    split_mode: str
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    X_all: np.ndarray | None = None
    y_all: np.ndarray | None = None

class BaseDataset(ABC):
    dataset_id: str
    subject_list: list[int]

    @abstractmethod
    def build_subject_cache(self, subject: int) -> CachedSubject:
        raise NotImplementedError
