from pathlib import Path
import json

from refshift.specs import load_dataset_spec
from refshift.preprocessing.montage import load_positions_csv, build_laplacian_knn, build_bipolar_nearest


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "artifacts" / "graphs"
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset_id in ["iv2a", "openbmi", "cho2017", "dreyer2023"]:
        spec = load_dataset_spec(repo_root, dataset_id)
        df = load_positions_csv(repo_root / spec.positions_artifact)
        lap = build_laplacian_knn(df, k=4)
        bip = build_bipolar_nearest(df)

        (out_root / f"{dataset_id}_laplacian_knn4.json").write_text(json.dumps(lap, indent=2))
        (out_root / f"{dataset_id}_bipolar_nearest.json").write_text(json.dumps(bip, indent=2))
        print(f"Rebuilt graph artifacts for {dataset_id}")


if __name__ == "__main__":
    main()
