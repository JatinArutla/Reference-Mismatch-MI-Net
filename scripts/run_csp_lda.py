
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from refshift.experiments.common import MODES, load_dataset_yaml, load_graphs, get_subject_data
from refshift.baselines.csp_lda import run_csp_lda
from refshift.utils.io import save_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['iv2a','openbmi','cho2017','dreyer2023'])
    p.add_argument('--cache_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--repo_root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--subjects', default='default')
    p.add_argument('--standardization', choices=['mechanistic','deployment'], default='mechanistic')
    args = p.parse_args()
    repo_root = Path(args.repo_root)
    ds_spec = load_dataset_yaml(repo_root, args.dataset)
    neighbor_map, partner_map = load_graphs(repo_root, ds_spec)
    subjects = list(ds_spec['subjects_default']) if args.subjects == 'default' else [int(x) for x in args.subjects.split(',') if x.strip()]
    rows=[]; mat=[]
    for train_mode in MODES:
        for test_mode in MODES:
            accs=[]
            for subj in subjects:
                Xtr,ytr,Xte,yte,ch_names = get_subject_data(args.cache_root, args.dataset, ds_spec, int(subj))
                metrics,_ = run_csp_lda(Xtr,ytr,Xte,yte,train_mode,test_mode,ch_names,neighbor_map,partner_map,args.standardization)
                rows.append({'subject': int(subj), 'train_mode': train_mode, 'test_mode': test_mode, **metrics})
                accs.append(metrics['acc'])
            mat.append({'train_mode': train_mode, 'test_mode': test_mode, 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs))})
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_csv(pd.DataFrame(rows), out/'metrics_subject.csv')
    save_csv(pd.DataFrame(mat), out/'matrix_subjectwise.csv')
    save_csv(pd.DataFrame(mat).pivot(index='train_mode', columns='test_mode', values='acc_mean').reset_index(), out/'matrix_mean.csv')

if __name__ == '__main__':
    main()
