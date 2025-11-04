from __future__ import annotations

from typing import List, Tuple, Optional

import math
import numpy as np

from .transforms import procrustes_2d, rotation_matrix_deg
from .features import CircleFeature, LFeature


def _wrap_angle_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return a


def ransac_rigid_from_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    iters: int = 300,
    tol: float = 0.08,
) -> Tuple[np.ndarray, float, List[Tuple[int, int]]]:
    """RANSAC estimate transform mapping src->dst from two-point hypotheses.

    Returns (t, theta_deg, inlier_pairs).
    """
    n1, n2 = len(src_pts), len(dst_pts)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 points in each set for RANSAC")

    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i1, i2 = rng.choice(n1, size=2, replace=False)
        j1, j2 = rng.choice(n2, size=2, replace=False)
        A = np.vstack([src_pts[i1], src_pts[i2]])
        B = np.vstack([dst_pts[j1], dst_pts[j2]])
        try:
            t_h, th_h = procrustes_2d(A, B)
        except Exception:
            continue
        R = rotation_matrix_deg(th_h)
        # associate by nearest neighbor under hypothesis
        mapped = (R @ src_pts.T).T + t_h
        pairs: List[Tuple[int, int]] = []
        used_j = set()
        for i, p in enumerate(mapped):
            d = np.linalg.norm(dst_pts - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used_j:
                pairs.append((i, j))
                used_j.add(j)
        if best is None or len(pairs) > len(best[2]):
            best = (t_h, th_h, pairs)
    assert best is not None
    # refine with inliers
    in_src = np.vstack([src_pts[i] for i, _ in best[2]]) if best[2] else src_pts[:2]
    in_dst = np.vstack([dst_pts[j] for _, j in best[2]]) if best[2] else dst_pts[:2]
    t_ref, th_ref = procrustes_2d(in_src, in_dst)
    return t_ref, th_ref, best[2]


def estimate_transform_circles(
    f1: List[CircleFeature], f2: List[CircleFeature]
) -> Tuple[np.ndarray, float, List[Tuple[int, int]]]:
    P1 = np.array([c.center for c in f1], dtype=float)
    P2 = np.array([c.center for c in f2], dtype=float)
    return ransac_rigid_from_points(P1, P2)


def _angle_from_vec(v: np.ndarray) -> float:
    return float(math.degrees(math.atan2(v[1], v[0])))


def estimate_transform_single_L_pair(s: LFeature, d: LFeature) -> Tuple[np.ndarray, float]:
    """Estimate transform from a single L pair using line orientations and intersection point.

    Choose the rotation aligning s.line1 to either of d's lines (considering 180° ambiguity),
    picking the orientation that also best aligns the second line. Translation aligns intersections.
    """
    # candidate dest directions
    s1 = s.line1[1]
    s2 = s.line2[1]
    d1 = d.line1[1]
    d2 = d.line2[1]

    def normang(a):
        return (a + 180.0) % 360.0 - 180.0

    ang_s1 = _angle_from_vec(s1)
    ang_s2 = _angle_from_vec(s2)
    cand = []
    for ang_d_primary in (_angle_from_vec(d1), _angle_from_vec(d2)):
        # align s1 -> d_primary (mod 180)
        base = normang(ang_d_primary - ang_s1)
        # evaluate how well s2 aligns to the other dest line
        ang_d_other = _angle_from_vec(d2) if ang_d_primary == _angle_from_vec(d1) else _angle_from_vec(d1)
        res = min(
            abs(normang(ang_d_other - (ang_s2 + base))),
            abs(normang(ang_d_other - (ang_s2 + base + 180.0))),
        )
        cand.append((res, base))
    cand.sort(key=lambda x: x[0])
    theta = cand[0][1]
    R = rotation_matrix_deg(theta)
    t = np.array(d.lc) - (R @ np.array(s.lc).reshape(2, 1)).reshape(2)
    return t, theta


def estimate_transform_single_L_pair_candidates(s: LFeature, d: LFeature) -> List[Tuple[np.ndarray, float]]:
    """Return both orientation candidates for a single L pair.

    There are typically two rotations that align the two unoriented lines: mapping s.line1->d.line1
    or s.line1->d.line2 (modulo 180°). This returns both (t, theta_deg) candidates.
    """
    s1 = s.line1[1]
    s2 = s.line2[1]
    d1 = d.line1[1]
    d2 = d.line2[1]
    ang_s1 = _angle_from_vec(s1)
    cand = []
    for ang_d_primary in (_angle_from_vec(d1), _angle_from_vec(d2)):
        base = (ang_d_primary - ang_s1 + 180.0) % 360.0 - 180.0
        R = rotation_matrix_deg(base)
        t = np.array(d.lc) - (R @ np.array(s.lc).reshape(2, 1)).reshape(2)
        cand.append((t, base))
    # ensure uniqueness
    uniq = []
    seen = set()
    for t, th in cand:
        key = (round(float(th), 6), round(float(t[0]), 6), round(float(t[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((t, th))
    return uniq


def estimate_transform_Ls(
    f1: List[LFeature], f2: List[LFeature], raw_pts1: Optional[np.ndarray] = None, raw_pts2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, List[Tuple[int, int]]]:
    """Estimate transform using L features.

    Strategy: try all pairs whose included angle is close (|ldeg diff| <= 10°),
    compute orientation-based transform from that single pair, then count inliers
    among all intersections. Choose the transform with most inliers and lowest residual.
    """
    P1 = np.array([l.lc for l in f1], dtype=float)
    P2 = np.array([l.lc for l in f2], dtype=float)
    def count_inliers(t, th):
        R = rotation_matrix_deg(th)
        mapped = (R @ P1.T).T + t
        pairs = []
        used = set()
        for i, p in enumerate(mapped):
            d = np.linalg.norm(P2 - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= 0.3 and j not in used:
                pairs.append((i, j, d[j]))
                used.add(j)
        mse = float(np.mean([pp[2] for pp in pairs])) if pairs else 1e9
        return pairs, mse

    def points_alignment_mse(t, th, center_s=None, center_d=None, radius=1.5):
        if raw_pts1 is None or raw_pts2 is None:
            return 0.0
        R = rotation_matrix_deg(th)
        if center_s is not None:
            c = np.array(center_s)
            idx = np.linalg.norm(raw_pts1 - c, axis=1) <= radius
            src = raw_pts1[idx]
        else:
            src = raw_pts1
        mapped = (R @ src.T).T + t
        if center_d is not None:
            cd = np.array(center_d)
            dst = raw_pts2[np.linalg.norm(raw_pts2 - cd, axis=1) <= radius]
        else:
            dst = raw_pts2
        dists = []
        for p in mapped:
            if len(dst) == 0:
                d = np.linalg.norm(raw_pts2 - p, axis=1)
            else:
                d = np.linalg.norm(dst - p, axis=1)
            dists.append(float(np.min(d)))
        d = np.array(dists)
        # robust: take median of the closest half
        k = max(1, len(d) // 2)
        return float(np.sort(d)[:k].mean())

    best = None
    for i, s in enumerate(f1):
        for j, d in enumerate(f2):
            if abs(s.ldeg - d.ldeg) > 10.0:
                continue
            t, th = estimate_transform_single_L_pair(s, d)
            pairs, mse = count_inliers(t, th)
            mse_pts = points_alignment_mse(t, th, center_s=s.lc, center_d=d.lc)
            key = (-len(pairs), round(mse,6), round(mse_pts,6))
            if best is None or key < best[0]:
                best = (key, t, th, [(i, j)] + [(ii, jj) for ii, jj, _ in pairs if (ii, jj) != (i, j)])
    if best is None:
        # fallback to intersection-only RANSAC if no angle-consistent pairs found
        if len(P1) >= 2 and len(P2) >= 2:
            t, th, pairs = ransac_rigid_from_points(P1, P2, tol=0.2)
            return t, th, pairs
        # worst-case: take closest intersection pair
        t, th = estimate_transform_single_L_pair(f1[0], f2[0])
        return t, th, [(0, 0)]
    return best[1], best[2], best[3]
