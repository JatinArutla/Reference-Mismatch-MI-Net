
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from refshift.experiments.common import MODES, load_dataset_yaml, load_graphs, get_subject_data
from refshift.training.supervised import TrainConfig, fit_jitter_atcnet, _prepare_fixed, predict_metrics
from refshift.utils.io import save_csv, save_yaml


def run_jitter(repo_root: Path, dataset_id: str, cache_root: str, out_dir: str, cfg: TrainConfig, train_modes=None, subjects=None):
    ds_spec = load_dataset_yaml(repo_root, dataset_id)
    subjects = subjects or list(ds_spec['subjects_default'])
    train_modes = train_modes or MODES
    neighbor_map, partner_map = load_graphs(repo_root, ds_spec)
    rows = []
    summary = []
    for test_mode in MODES:
        vals = []
        for subj in subjects:
            Xtr, ytr, Xte, yte, ch_names = get_subject_data(cache_root, dataset_id, ds_spec, int(subj))
            n_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
            model = fit_jitter_atcnet(Xtr, ytr, Xte, yte, n_classes, cfg, ch_names, neighbor_map, partner_map, train_modes)
            # evaluate under a fixed test mode
            _, Xte_eval = _prepare_fixed(Xtr[:1], Xte, 'native', test_mode, ch_names, neighbor_map, partner_map, cfg.standardization)  # dummy train part ignored
            metrics, _ = predict_metrics(model, Xte_eval, yte)
            vals.append(metrics['acc'])
            rows.append({'subject': int(subj), 'test_mode': test_mode, **metrics})
        summary.append({'test_mode': test_mode, 'acc_mean': float(np.mean(vals)), 'acc_std': float(np.std(vals))})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    save_csv(pd.DataFrame(rows), out/'metrics_subject.csv')
    save_csv(pd.DataFrame(summary), out/'metrics_summary.csv')
    save_yaml(cfg.__dict__, out/'config.yaml')
