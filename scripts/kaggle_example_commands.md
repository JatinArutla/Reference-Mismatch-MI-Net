
# Kaggle example commands

```python
import os, pathlib, subprocess
os.chdir('/kaggle/working')
repo_dir = pathlib.Path('refshift_full_repo_v0')
```

```bash
python scripts/build_cache.py --dataset iv2a --data_root /kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014 --cache_root /kaggle/working/cache
python scripts/run_benchmark.py --dataset iv2a --cache_root /kaggle/working/cache --out_dir /kaggle/working/results/iv2a_fixed --epochs 200 --seed 1
python scripts/run_jitter.py --dataset iv2a --cache_root /kaggle/working/cache --out_dir /kaggle/working/results/iv2a_jitter --epochs 200 --seed 1
```
