from __future__ import annotations

import numpy as np

deg2rad = np.pi / 180.0


def rotation_matrix_deg(theta_deg: float) -> np.ndarray:
    th = theta_deg * deg2rad
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(X: np.ndarray, t: np.ndarray, theta_deg: float) -> np.ndarray:
    R = rotation_matrix_deg(theta_deg)
    return (R @ X.T).T + t


def invert_transform(t: np.ndarray, theta_deg: float) -> tuple[np.ndarray, float]:
    R_inv = rotation_matrix_deg(-theta_deg)
    t_inv = -(R_inv @ t.reshape(2, 1)).reshape(2)
    return t_inv, -theta_deg


def compose_transforms(t_ab: np.ndarray, theta_ab: float, t_bc: np.ndarray, theta_bc: float) -> tuple[np.ndarray, float]:
    """Compose A->B then B->C to get A->C.

    p_c = R_bc (R_ab p_a + t_ab) + t_bc = R_bc R_ab p_a + (R_bc t_ab + t_bc)
    """
    R_bc = rotation_matrix_deg(theta_bc)
    t_ac = (R_bc @ t_ab.reshape(2, 1)).reshape(2) + t_bc
    theta_ac = theta_ab + theta_bc
    return t_ac, theta_ac


def procrustes_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate rigid transform (t, theta_deg) mapping src->dst from paired points.

    Uses SVD on centered coordinates. Returns translation t and rotation angle in degrees.
    """
    assert src.shape == dst.shape and src.shape[0] >= 2
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    X = src - c_src
    Y = dst - c_dst
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    t = c_dst - (R @ c_src)
    return t, theta


def compute_to_ref(
    adjacency: np.ndarray,
    edge_transforms: dict[tuple[int, int], tuple[np.ndarray, float]],
    root: int = 0,
) -> list[tuple[np.ndarray, float]]:
    """Compute transforms from each node to the reference node using BFS.

    Args:
        adjacency: NxN adjacency matrix (nonzero indicates edge, assumed undirected).
        edge_transforms: mapping (i,j)->(t_ij, th_ij) giving transform from i to j.
        root: reference node index (default 0).

    Returns:
        List of (t_i1, theta_i1) mapping node i to root.
    """
    N = adjacency.shape[0]
    to_ref: list[tuple[np.ndarray, float] | None] = [None] * N
    to_ref[root] = (np.zeros(2, dtype=float), 0.0)
    from collections import deque
    q = deque([root])
    while q:
        u = q.popleft()
        t_u1, th_u1 = to_ref[u]  # type: ignore
        for v in range(N):
            if v == u or adjacency[u, v] == 0 or to_ref[v] is not None:
                continue
            # find transform for edge u<->v
            if (u, v) in edge_transforms:
                t_uv, th_uv = edge_transforms[(u, v)]
                t_vu, th_vu = invert_transform(t_uv, th_uv)
                t_v1, th_v1 = compose_transforms(t_vu, th_vu, t_u1, th_u1)
            elif (v, u) in edge_transforms:
                t_vu, th_vu = edge_transforms[(v, u)]
                t_v1, th_v1 = compose_transforms(t_vu, th_vu, t_u1, th_u1)
            else:
                raise KeyError(f"Missing edge transform for ({u},{v}) or ({v},{u})")
            to_ref[v] = (t_v1, th_v1)
            q.append(v)
    # type: ignore
    return to_ref  # type: ignore
