
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from refshift.experiments.common import FAMILIES, load_dataset_yaml, load_graphs, get_subject_data, print_acc_table
from refshift.training.supervised import TrainConfig, fit_jitter_atcnet, _prepare_fixed, predict_metrics
from refshift.utils.io import save_csv, save_yaml


def run_lofo(repo_root: Path, dataset_id: str, cache_root: str, out_dir: str, cfg: TrainConfig, subjects=None):
    ds_spec = load_dataset_yaml(repo_root, dataset_id)
    subjects = subjects or list(ds_spec['subjects_default'])
    neighbor_map, partner_map = load_graphs(repo_root, ds_spec)
    rows = []
    for heldout_family, heldout_modes in FAMILIES.items():
        print(f'\n=== Training LOFO with held-out family: {heldout_family} ===')
        train_modes = [m for fam, modes in FAMILIES.items() if fam != heldout_family for m in modes]
        for subj in subjects:
            Xtr, ytr, Xte, yte, ch_names = get_subject_data(cache_root, dataset_id, ds_spec, int(subj))
            n_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
            model = fit_jitter_atcnet(Xtr, ytr, Xte, yte, n_classes, cfg, ch_names, neighbor_map, partner_map, train_modes)
            mode_metrics = []
            for tm in heldout_modes:
                _, Xte_eval = _prepare_fixed(Xtr[:1], Xte, 'native', tm, ch_names, neighbor_map, partner_map, cfg.standardization)
                metrics, _ = predict_metrics(model, Xte_eval, yte)
                mode_metrics.append(metrics['acc'])
            rows.append({'subject': int(subj), 'heldout_family': heldout_family, 'acc': float(np.mean(mode_metrics))})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    save_csv(df, out/'metrics_subject.csv')
    summary_df = df.groupby('heldout_family', as_index=False)['acc'].mean().rename(columns={'heldout_family':'train_mode','acc':'acc_mean'})
    save_csv(summary_df.rename(columns={'train_mode':'heldout_family','acc_mean':'acc'}), out/'metrics_summary.csv')
    save_yaml(cfg.__dict__, out/'config.yaml')
    print('\n=== LOFO summary ===')
    print_acc_table(summary_df.assign(test_mode='heldout'), row_key='train_mode', col_key='test_mode', value_key='acc_mean', row_order=list(FAMILIES.keys()), col_order=['heldout'], title='Averaged accuracy (%)')
