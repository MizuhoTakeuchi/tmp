import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from map_composer import load_adjacency, infer_dataset_type  # type: ignore


def run_case(folder: str, expected: str) -> None:
    base = Path(folder)
    maps = [str(base / f"map{i}.csv") for i in (1, 2, 3)]
    adj = load_adjacency(str(Path("data/adj.csv")))
    detected = infer_dataset_type(maps, adj)
    print(f"{folder}: detected={detected}, expected={expected}")
    assert (
        detected == expected
    ), f"Type mismatch for {folder}: detected={detected}, expected={expected}"


if __name__ == "__main__":
    run_case("data/circle", "circle")
    run_case("data/L", "L")
    print("All tests passed.")
