
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from refshift.utils.cache import load_cached_subject
from refshift.utils.io import load_yaml
from refshift.preprocessing.montage import load_graph_json

MODES = ['native','car','laplacian','bipolar','gs','median']
FAMILIES = {
    'native': ['native'],
    'global': ['car','gs','median'],
    'local': ['laplacian','bipolar'],
}


def repo_root_from_file(file: str) -> Path:
    return Path(file).resolve().parents[3]


def load_dataset_yaml(repo_root: Path, dataset_id: str):
    return load_yaml(repo_root / 'specs' / 'datasets' / f'{dataset_id}.yaml')


def load_graphs(repo_root: Path, dataset_spec: dict):
    lap = load_graph_json(repo_root / dataset_spec['laplacian_graph_artifact'])
    bip = load_graph_json(repo_root / dataset_spec['bipolar_graph_artifact'])
    return lap, bip


def get_subject_data(cache_root: str, dataset_id: str, dataset_spec: dict, subject: int):
    item = load_cached_subject(cache_root, dataset_id, subject)
    if item is None:
        raise FileNotFoundError(f'No cached subject found for {dataset_id} subject {subject}. Run build_cache.py first.')
    if dataset_spec['split_mode'] == 'session':
        return item.X_train, item.y_train, item.X_test, item.y_test, item.channel_names
    X_all, y_all = item.X_all, item.y_all
    sss = StratifiedShuffleSplit(n_splits=1, train_size=float(dataset_spec.get('train_frac', 0.8)), random_state=int(dataset_spec.get('split_seed', 1)))
    tr_idx, te_idx = next(sss.split(X_all, y_all))
    return X_all[tr_idx], y_all[tr_idx], X_all[te_idx], y_all[te_idx], item.channel_names



def format_acc_table(df, row_key, col_key, value_key='acc_mean', row_order=None, col_order=None, title='Averaged accuracy (%)'):
    import pandas as pd
    row_order = list(row_order) if row_order is not None else list(dict.fromkeys(df[row_key].tolist()))
    col_order = list(col_order) if col_order is not None else list(dict.fromkeys(df[col_key].tolist()))
    piv = df.pivot(index=row_key, columns=col_key, values=value_key)
    piv = piv.reindex(index=row_order, columns=col_order)
    row_name = f"{row_key}\\{col_key}"
    first_w = max(len(row_name), max(len(str(x)) for x in row_order)) + 2
    col_ws = {c: max(len(str(c)), 10) + 2 for c in col_order}
    lines = []
    lines.append(f'{title}:')
    header = row_name.ljust(first_w) + ''.join(str(c).ljust(col_ws[c]) for c in col_order)
    lines.append(header)
    lines.append('-' * len(header))
    for r in row_order:
        line = str(r).ljust(first_w)
        for c in col_order:
            val = piv.loc[r, c] if (r in piv.index and c in piv.columns) else float('nan')
            sval = 'nan' if pd.isna(val) else f'{float(val):.2f}'
            line += sval.ljust(col_ws[c])
        lines.append(line)
    return '\n'.join(lines)


def print_acc_table(df, row_key, col_key, value_key='acc_mean', row_order=None, col_order=None, title='Averaged accuracy (%)'):
    print(format_acc_table(df, row_key=row_key, col_key=col_key, value_key=value_key, row_order=row_order, col_order=col_order, title=title))
