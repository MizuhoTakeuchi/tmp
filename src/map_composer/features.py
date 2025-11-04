from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np

from .clustering import epsilon_connected_components


@dataclass
class CircleFeature:
    center: Tuple[float, float]
    radius: float
    residual: float


@dataclass
class LFeature:
    lc: Tuple[float, float]
    l1: float
    l2: float
    ldeg: float
    line1: Tuple[np.ndarray, np.ndarray]  # (p0, dir)
    line2: Tuple[np.ndarray, np.ndarray]


def _fit_circle_kasa(X: np.ndarray) -> Tuple[float, float, float]:
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
    return float(cx), float(cy), float(math.sqrt(R2))


def _circle_residual_norm(X: np.ndarray, cx: float, cy: float, R: float) -> float:
    r = np.hypot(X[:, 0] - cx, X[:, 1] - cy)
    err = np.abs(r - R)
    return float(err.mean() / max(R, 1e-9))


def extract_circle_features(
    maps: List[np.ndarray], eps: float = 0.1, min_samples: int = 5, residual_th: float = 0.02
) -> List[List[CircleFeature]]:
    """Extract circle centers per map via clustering + circle fit.

    Returns a list per map; each contains CircleFeature for clusters whose
    normalized residual is below residual_th.
    """
    all_features: List[List[CircleFeature]] = []
    for pts in maps:
        labels, n_clusters = epsilon_connected_components(pts, eps=eps, min_samples=min_samples)
        feats: List[CircleFeature] = []
        for cid in range(n_clusters):
            Xc = pts[labels == cid]
            if len(Xc) < max(6, min_samples):
                continue
            try:
                cx, cy, R = _fit_circle_kasa(Xc)
            except Exception:
                continue
            resn = _circle_residual_norm(Xc, cx, cy, R)
            if resn <= residual_th:
                feats.append(CircleFeature(center=(cx, cy), radius=R, residual=resn))
        all_features.append(feats)
    return all_features


def _fit_line_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a line by PCA; returns (mean point p0, direction unit vector d, projections t).

    t are scalars such that p = p0 + t*d for each input p.
    """
    p0 = X.mean(axis=0)
    Y = X - p0
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    d = Vt[0]
    d = d / np.linalg.norm(d)
    t = Y @ d  # projections
    return p0, d, t


def _lines_intersection(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> Tuple[bool, np.ndarray]:
    """Compute intersection point of two infinite lines p0+t*d0 and p1+s*d1.

    Returns (ok, point). If nearly parallel, ok=False and point is midpoint of closest approach.
    """
    A = np.stack([d0, -d1], axis=1)  # 2x2
    b = (p1 - p0)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        # parallel/near-parallel: return midpoint between closest points on the lines
        # project vector between points onto normal of d0
        n = np.array([-d0[1], d0[0]])
        t = (b @ n) / (d1 @ n + 1e-12)
        q1 = p1 + t * d1
        q0 = p0 + ((q1 - p0) @ d0) * d0
        return False, (q0 + q1) / 2.0
    ts = np.linalg.solve(A, b)
    t0 = ts[0]
    pt = p0 + t0 * d0
    return True, pt


def extract_L_features(
    maps: List[np.ndarray],
    eps: float = 0.12,
    min_samples: int = 8,
    angle_min_deg: float = 20.0,
    endpoint_frac_th: float = 0.5,
    min_length: float = 0.2,
) -> List[List[LFeature]]:
    """Extract L-shape features per map by first clustering entire L groups, then splitting into two lines.

    For each proximity cluster (DBSCAN-like), we fit two dominant lines via
    iterative assignment (closest-line labeling) and PCA refits. The
    intersection must be near the end of both segments to be accepted as an L.
    """
    results: List[List[LFeature]] = []
    for pts in maps:
        labels, n_clusters = epsilon_connected_components(pts, eps=eps, min_samples=min_samples)
        feats_for_map: List[LFeature] = []
        for cid in range(n_clusters):
            X = pts[labels == cid]
            if len(X) < max(20, min_samples * 2):
                continue
            # initial line1 by PCA
            p01, d1, t1 = _fit_line_pca(X)
            # k-means (k=2) on absolute perpendicular distance to line1
            d_perp = np.linalg.norm((X - p01) - np.outer(t1, d1), axis=1)
            m1, m2 = float(np.percentile(d_perp, 10)), float(np.percentile(d_perp, 90))
            for _ in range(10):
                if abs(m1 - m2) < 1e-6:
                    break
                assign1 = np.abs(d_perp - m1) <= np.abs(d_perp - m2)
                if assign1.sum() == 0 or (~assign1).sum() == 0:
                    break
                m1 = float(d_perp[assign1].mean())
                m2 = float(d_perp[~assign1].mean())
            # smaller-mean cluster -> line1; other -> line2
            mask1 = assign1 if m1 <= m2 else ~assign1
            if mask1.sum() < 5 or (~mask1).sum() < 5:
                continue
            # initialize second line perpendicular to the first at same center
            p02 = p01.copy()
            d2 = np.array([-d1[1], d1[0]])
            # refine: assign to nearest of the two lines, refit (few iters)
            for _ in range(6):
                # distances to both lines
                def point_line_dist(P, p0, d):
                    u = P - p0
                    proj = (u @ d)[:, None] * d
                    return np.linalg.norm(u - proj, axis=1)
                dA = point_line_dist(X, p01, d1)
                dB = point_line_dist(X, p02, d2)
                assign1 = dA <= dB
                if assign1.sum() < 5 or (~assign1).sum() < 5:
                    break
                p01, d1, t1 = _fit_line_pca(X[assign1])
                p02, d2, t2 = _fit_line_pca(X[~assign1])
            # compute extents and intersection
            t1 = (X - p01) @ d1
            t2 = (X - p02) @ d2
            t1min, t1max = float(np.min(t1)), float(np.max(t1))
            t2min, t2max = float(np.min(t2)), float(np.max(t2))
            L1 = float(t1max - t1min); L2 = float(t2max - t2min)
            if L1 < min_length or L2 < min_length:
                continue
            _, lc = _lines_intersection(p01, d1, p02, d2)
            # intersection location relative to extents
            t1_lc = float((lc - p01) @ d1)
            t2_lc = float((lc - p02) @ d2)
            def endpoint_frac(t, tmin, tmax):
                L = max(1e-9, tmax - tmin)
                return min(abs(t - tmin), abs(tmax - t)) / L
            e1 = endpoint_frac(t1_lc, t1min, t1max)
            e2 = endpoint_frac(t2_lc, t2min, t2max)
            ang = math.degrees(math.acos(abs(float(np.clip(d1 @ d2, -1.0, 1.0)))))
            if ang < angle_min_deg or e1 > endpoint_frac_th or e2 > endpoint_frac_th:
                continue
            l1 = float(max(abs(t1max - t1_lc), abs(t1_lc - t1min)))
            l2 = float(max(abs(t2max - t2_lc), abs(t2_lc - t2min)))
            # enforce line1 to be the longer arm to stabilize orientation mapping across maps
            if l2 > l1:
                l1, l2 = l2, l1
                p01, p02 = p02, p01
                d1, d2 = d2, d1
            feats_for_map.append(LFeature(lc=(float(lc[0]), float(lc[1])), l1=l1, l2=l2, ldeg=ang, line1=(p01, d1), line2=(p02, d2)))
        results.append(feats_for_map)
    return results
