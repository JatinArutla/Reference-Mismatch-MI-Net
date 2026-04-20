
from __future__ import annotations
from pathlib import Path
from refshift.utils.io import load_yaml


def load_dataset_spec(repo_root: str | Path, dataset_id: str):
    return load_yaml(Path(repo_root) / 'specs' / 'datasets' / f'{dataset_id}.yaml')
