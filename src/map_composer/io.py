from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_maps(map_files: List[str]) -> List[np.ndarray]:
    """Load CSV maps with header 'x,y' into arrays of shape (N,2).

    Args:
        map_files: List of CSV file paths.

    Returns:
        List of float64 arrays (N_i, 2).
    """
    arrays: List[np.ndarray] = []
    for p in map_files:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {p}")
        arr = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float)
        # Support both structured and regular arrays
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            X = np.column_stack([arr["x"], arr["y"]]).astype(float)
        else:
            X = np.asarray(arr, dtype=float)
            if X.ndim == 1 and X.size == 2:
                X = X.reshape(1, 2)
        arrays.append(X)
    return arrays


def load_adjacency(path: str) -> np.ndarray:
    """Load adjacency matrix from CSV (no header).

    Args:
        path: CSV file path.

    Returns:
        Square numpy array of ints.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Adjacency file not found: {path}")
    mat = np.genfromtxt(str(p), delimiter=",", dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    return mat.astype(int)
