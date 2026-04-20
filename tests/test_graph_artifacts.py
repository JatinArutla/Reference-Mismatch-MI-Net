from pathlib import Path
import json


def test_graph_files_exist():
    repo_root = Path(__file__).resolve().parents[1]
    for dataset_id in ["iv2a", "openbmi", "cho2017", "dreyer2023"]:
        lap = repo_root / "artifacts" / "graphs" / f"{dataset_id}_laplacian_knn4.json"
        bip = repo_root / "artifacts" / "graphs" / f"{dataset_id}_bipolar_nearest.json"
        assert lap.exists()
        assert bip.exists()
        lap_obj = json.loads(lap.read_text())
        bip_obj = json.loads(bip.read_text())
        assert len(lap_obj) > 0
        assert len(bip_obj) > 0
