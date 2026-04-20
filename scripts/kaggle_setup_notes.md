# Kaggle setup notes

## Option A: clone from your GitHub repo
Push this folder to your GitHub repo locally first, then in Kaggle:

```python
import os, pathlib, subprocess
os.chdir("/kaggle/working")

repo_dir = pathlib.Path("refshift_full_repo_v1")
if not repo_dir.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/<YOUR_USER>/<YOUR_REPO>.git", str(repo_dir)],
        check=True
    )

os.chdir(repo_dir)
print("CWD:", os.getcwd())
```

## Option B: upload the zip as a Kaggle dataset/input
If you upload `refshift_full_repo_v1.zip` as a Kaggle Dataset or notebook input:

```python
import os, pathlib, zipfile
os.chdir("/kaggle/working")

zip_path = pathlib.Path("/kaggle/input/<YOUR_DATASET_NAME>/refshift_full_repo_v1.zip")
repo_dir = pathlib.Path("/kaggle/working/refshift_full_repo_v1")

if not repo_dir.exists():
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/kaggle/working")

os.chdir(repo_dir)
print("CWD:", os.getcwd())
```
