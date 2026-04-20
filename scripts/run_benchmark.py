
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from refshift.training.supervised import TrainConfig
from refshift.experiments.benchmark import run_fixed_6x6


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['iv2a','openbmi','cho2017','dreyer2023'])
    p.add_argument('--cache_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--repo_root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--subjects', default='default')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--standardization', choices=['mechanistic','deployment'], default='mechanistic')
    args = p.parse_args()
    subjects = None if args.subjects == 'default' else [int(x) for x in args.subjects.split(',') if x.strip()]
    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, standardization=args.standardization)
    run_fixed_6x6(Path(args.repo_root), args.dataset, args.cache_root, args.out_dir, cfg, subjects)

if __name__ == '__main__':
    main()
