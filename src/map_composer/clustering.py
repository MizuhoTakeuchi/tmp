from __future__ import annotations

from typing import Tuple

import numpy as np


def epsilon_connected_components(
    X: np.ndarray, eps: float = 0.1, min_samples: int = 5
) -> Tuple[np.ndarray, int]:
    """Simple DBSCAN-like clustering using epsilon-neighborhood connectivity.

    Args:
        X: Points (N,2).
        eps: Neighborhood radius.
        min_samples: Minimum neighbors to start/expand a cluster.

    Returns:
        labels: Cluster labels for each point, -1 for noise.
        n_clusters: Number of clusters (labels in [0, n_clusters-1]).
    """
    if X.size == 0:
        return np.empty((0,), dtype=int), 0
    n = len(X)
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    # distance matrix (O(N^2)) – fine for small N
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(D[i] <= eps)[0]
        if neighbors.size < min_samples:
            labels[i] = -1
            continue
        labels[neighbors] = cid
        stack = list(neighbors)
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            n2 = np.where(D[j] <= eps)[0]
            if n2.size >= min_samples:
                new = [k for k in n2 if labels[k] != cid]
                labels[new] = cid
                stack.extend(new)
        cid += 1
    return labels, cid
