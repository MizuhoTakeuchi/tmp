from __future__ import annotations

from typing import List, Tuple

import math
import numpy as np

from .clustering import epsilon_connected_components


def _fit_circle_kasa(X: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic circle fit (Kasa method): returns (cx, cy, R).

    Raises ValueError if fit fails.
    """
    if len(X) < 3:
        raise ValueError("Not enough points to fit a circle")
    x, y = X[:, 0], X[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x**2 + y**2)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    A_, B_, C_ = coeffs
    cx, cy = -A_ / 2.0, -B_ / 2.0
    R2 = (A_**2 + B_**2) / 4.0 - C_
    if R2 <= 0:
        raise ValueError("Non-positive radius squared from circle fit")
    R = math.sqrt(R2)
    return float(cx), float(cy), float(R)


def _circle_normalized_residual(X: np.ndarray) -> float:
    """Mean absolute radial residual normalized by radius."""
    try:
        cx, cy, R = _fit_circle_kasa(X)
    except Exception:
        return float("inf")
    r = np.hypot(X[:, 0] - cx, X[:, 1] - cy)
    err = np.abs(r - R)
    return float(err.mean() / max(R, 1e-9))


def infer_dataset_type(
    map_files: List[str], adjacency: np.ndarray, eps: float = 0.1, min_samples: int = 5
) -> str:
    """Infer global point cloud type ("circle" or "L") from maps.

    The requirement guarantees only one type per dataset. We cluster all points
    using epsilon-connectivity and judge per-cluster by circle fit residual.

    Args:
        map_files: CSV paths with header x,y.
        adjacency: Adjacency matrix (unused for type inference but validated).
        eps: Epsilon for clustering.
        min_samples: Min points to form a cluster.

    Returns:
        "circle" or "L".
    """
    # Lazy import to avoid cycle
    from .io import load_maps

    maps = load_maps(map_files)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square matrix")

    all_pts = np.vstack(maps) if len(maps) else np.empty((0, 2))
    if len(all_pts) == 0:
        raise ValueError("No points loaded from maps")

    labels, n_clusters = epsilon_connected_components(all_pts, eps=eps, min_samples=min_samples)
    # Evaluate each cluster; treat noise (-1) as its own tiny clusters only if abundant
    residuals: List[float] = []
    for cid in range(n_clusters):
        Xc = all_pts[labels == cid]
        if len(Xc) < max(3, min_samples):
            continue
        residuals.append(_circle_normalized_residual(Xc))

    if not residuals:
        # Fallback: try whole set
        res = _circle_normalized_residual(all_pts)
        return "circle" if res < 0.02 else "L"

    # Decide by majority of clusters that are "circular enough"
    circular_flags = [res < 0.02 for res in residuals]
    circular_ratio = sum(circular_flags) / len(circular_flags)
    return "circle" if circular_ratio >= 0.5 else "L"
