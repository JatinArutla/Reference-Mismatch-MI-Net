from pathlib import Path
import json

from refshift.specs import load_dataset_spec


def main():
    repo_root = Path(__file__).resolve().parents[1]
    for dataset_id in ["iv2a", "openbmi", "cho2017", "dreyer2023"]:
        spec = load_dataset_spec(repo_root, dataset_id)
        print("=" * 80)
        print(dataset_id)
        print(json.dumps(spec.raw, indent=2)[:2000])


if __name__ == "__main__":
    main()
