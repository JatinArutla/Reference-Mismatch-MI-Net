
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import save_json, load_json, save_npz
from refshift.datasets.base import CachedSubject


def cache_path(cache_root, dataset_id: str, subject: int):
    root = Path(cache_root) / dataset_id / f'subject_{subject:03d}'
    return root


def save_cached_subject(cache_root, item: CachedSubject):
    root = cache_path(cache_root, item.dataset_id, item.subject)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        'dataset_id': item.dataset_id,
        'subject': item.subject,
        'channel_names': item.channel_names,
        'sfreq': item.sfreq,
        'split_mode': item.split_mode,
    }
    save_json(meta, root/'meta.json')
    arrays = {}
    for k in ['X_train','y_train','X_test','y_test','X_all','y_all']:
        v = getattr(item, k)
        if v is not None:
            arrays[k] = v
    save_npz(root/'data.npz', **arrays)


def load_cached_subject(cache_root, dataset_id: str, subject: int) -> CachedSubject | None:
    root = cache_path(cache_root, dataset_id, subject)
    meta_p = root/'meta.json'
    data_p = root/'data.npz'
    if not (meta_p.exists() and data_p.exists()):
        return None
    meta = load_json(meta_p)
    d = np.load(data_p, allow_pickle=False)
    item = CachedSubject(
        dataset_id=meta['dataset_id'], subject=meta['subject'], channel_names=meta['channel_names'],
        sfreq=float(meta['sfreq']), split_mode=meta['split_mode'],
        X_train=d['X_train'] if 'X_train' in d else None,
        y_train=d['y_train'] if 'y_train' in d else None,
        X_test=d['X_test'] if 'X_test' in d else None,
        y_test=d['y_test'] if 'y_test' in d else None,
        X_all=d['X_all'] if 'X_all' in d else None,
        y_all=d['y_all'] if 'y_all' in d else None,
    )
    return item
