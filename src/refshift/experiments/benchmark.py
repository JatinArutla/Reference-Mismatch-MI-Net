
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from refshift.experiments.common import MODES, load_dataset_yaml, load_graphs, get_subject_data
from refshift.training.supervised import TrainConfig, _prepare_fixed, fit_fixed_atcnet, predict_metrics
from refshift.utils.io import save_csv, save_json, save_yaml


def run_fixed_6x6(repo_root: Path, dataset_id: str, cache_root: str, out_dir: str, cfg: TrainConfig, subjects: list[int] | None = None):
    ds_spec = load_dataset_yaml(repo_root, dataset_id)
    subjects = subjects or list(ds_spec['subjects_default'])
    neighbor_map, partner_map = load_graphs(repo_root, ds_spec)
    rows = []
    matrix_by_seed = []
    for train_mode in MODES:
        for test_mode in MODES:
            vals = []
            for subj in subjects:
                Xtr, ytr, Xte, yte, ch_names = get_subject_data(cache_root, dataset_id, ds_spec, int(subj))
                n_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
                Xtr_m, Xte_m = _prepare_fixed(Xtr, Xte, train_mode, test_mode, ch_names, neighbor_map, partner_map, cfg.standardization)
                model = fit_fixed_atcnet(Xtr_m, ytr, Xte_m, yte, n_classes, cfg)
                metrics, _ = predict_metrics(model, Xte_m, yte)
                vals.append(metrics['acc'])
                rows.append({'subject': int(subj), 'train_mode': train_mode, 'test_mode': test_mode, **metrics})
            matrix_by_seed.append({'train_mode': train_mode, 'test_mode': test_mode, 'acc_mean': float(np.mean(vals)), 'acc_std': float(np.std(vals))})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(pd.DataFrame(rows), out/'metrics_subject.csv')
    save_csv(pd.DataFrame(matrix_by_seed), out/'matrix_subjectwise.csv')
    piv = pd.DataFrame(matrix_by_seed).pivot(index='train_mode', columns='test_mode', values='acc_mean').reset_index()
    save_csv(piv, out/'matrix_mean.csv')
    save_yaml(cfg.__dict__, out/'config.yaml')
