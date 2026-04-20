
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from refshift.datasets.registry import get_dataset
from refshift.utils.cache import save_cached_subject
from refshift.utils.io import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['iv2a','openbmi','cho2017','dreyer2023'])
    p.add_argument('--data_root', required=True)
    p.add_argument('--cache_root', required=True)
    p.add_argument('--subjects', default='default')
    p.add_argument('--repo_root', default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args()
    ds = get_dataset(args.repo_root, args.dataset, args.data_root)
    subjects = ds.subject_list if args.subjects == 'default' else [int(x) for x in args.subjects.split(',') if x.strip()]
    for s in subjects:
        item = ds.build_subject_cache(int(s))
        save_cached_subject(args.cache_root, item)
        print(f'cached {args.dataset} subject {s}')

if __name__ == '__main__':
    main()
