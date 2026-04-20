
from __future__ import annotations
from pathlib import Path
from refshift.utils.io import load_yaml
from .iv2a import IV2ADataset
from .openbmi import OpenBMIDataset
from .cho2017 import Cho2017Dataset
from .dreyer2023 import Dreyer2023Dataset


def load_dataset_spec(repo_root: str | Path, dataset_id: str):
    return load_yaml(Path(repo_root) / 'specs' / 'datasets' / f'{dataset_id}.yaml')


def get_dataset(repo_root: str | Path, dataset_id: str, data_root: str):
    spec = load_dataset_spec(repo_root, dataset_id)
    if dataset_id == 'iv2a':
        return IV2ADataset(data_root=data_root, spec=spec)
    if dataset_id == 'openbmi':
        return OpenBMIDataset(data_root=data_root, spec=spec)
    if dataset_id == 'cho2017':
        return Cho2017Dataset(data_root=data_root, spec=spec)
    if dataset_id == 'dreyer2023':
        return Dreyer2023Dataset(data_root=data_root, spec=spec)
    raise ValueError(dataset_id)
