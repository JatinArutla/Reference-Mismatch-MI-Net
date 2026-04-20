
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy
from refshift.experiments.common import load_dataset_yaml, load_graphs, get_subject_data
from refshift.training.supervised import TrainConfig, _prepare_fixed, fit_fixed_atcnet, predict_metrics
from refshift.utils.io import save_csv, save_yaml
from refshift.preprocessing.filters import bandpass_filter_trials
from refshift.preprocessing.standardization import standardize_trials_instance, fit_train_standardizer, apply_train_standardizer


def run_controls(repo_root: Path, dataset_id: str, cache_root: str, out_dir: str, cfg: TrainConfig, subjects=None):
    ds_spec = load_dataset_yaml(repo_root, dataset_id)
    subjects = subjects or list(ds_spec['subjects_default'])
    neighbor_map, partner_map = load_graphs(repo_root, ds_spec)
    rows = []
    for subj in subjects:
        Xtr, ytr, Xte, yte, ch_names = get_subject_data(cache_root, dataset_id, ds_spec, int(subj))
        n_classes = int(len(np.unique(np.concatenate([ytr, yte]))))
        # Reference mismatch native->car
        Xtr_a, Xte_a = _prepare_fixed(Xtr, Xte, 'native', 'car', ch_names, neighbor_map, partner_map, cfg.standardization)
        model = fit_fixed_atcnet(Xtr_a, ytr, Xte_a, yte, n_classes, cfg)
        m_ref, _ = predict_metrics(model, Xte_a, yte)
        rows.append({'subject': int(subj), 'control': 'reference_native_to_car', **m_ref})
        # Bandpass mismatch: train on existing 8-32, test 4-40
        fs = float(ds_spec['target_sfreq_hz'])
        Xte_bw = bandpass_filter_trials(Xte, fs, (4.0, 40.0))
        Xtr_b, Xte_b = _prepare_fixed(Xtr, Xte_bw, 'native', 'native', ch_names, neighbor_map, partner_map, cfg.standardization)
        model2 = fit_fixed_atcnet(Xtr_b, ytr, Xte_b, yte, n_classes, cfg)
        m_band, _ = predict_metrics(model2, Xte_b, yte)
        rows.append({'subject': int(subj), 'control': 'bandpass_8_32_to_4_40', **m_band})
        # Norm mismatch: train deployment standardization, test none-like by bypassing
        Xtr_r = Xtr.copy(); Xte_r = Xte.copy()
        mu, sd = fit_train_standardizer(Xtr_r)
        Xtr_r = apply_train_standardizer(Xtr_r, mu, sd)[:,None,:,:]
        Xte_r = Xte_r[:,None,:,:]
        model3 = fit_fixed_atcnet(Xtr_r, ytr, Xte_r, yte, n_classes, cfg)
        m_norm, _ = predict_metrics(model3, Xte_r, yte)
        rows.append({'subject': int(subj), 'control': 'normalization_train_to_none', **m_norm})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    save_csv(df, out/'metrics_subject.csv')
    save_csv(df.groupby('control', as_index=False)['acc'].mean(), out/'control_summary.csv')
    save_yaml(cfg.__dict__, out/'config.yaml')
