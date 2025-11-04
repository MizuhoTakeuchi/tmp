import os
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from map_composer import (
    load_maps,
    load_adjacency,
    infer_dataset_type,
    extract_circle_features,
    extract_L_features,
)
from map_composer.matching import (
    estimate_transform_circles,
    estimate_transform_Ls,
    estimate_transform_single_L_pair_candidates,
)
from map_composer.transforms import (
    transform_points,
    compose_transforms,
    invert_transform,
    compute_to_ref,
)


def estimate_for_folder(folder: str):
    base = Path(folder)
    map_paths = [str(base / f"map{i}.csv") for i in (1, 2, 3)]
    maps = load_maps(map_paths)
    adj = load_adjacency("data/adj.csv")
    dtype = infer_dataset_type(map_paths, adj)
    print(f"Dataset {folder} detected type: {dtype}")

    if dtype == "circle":
        feats = extract_circle_features(maps)
        estimator = estimate_transform_circles
        tol = 0.05
    else:
        feats = extract_L_features(maps)
        def estimator(f1, f2, i=None, j=None):
            return estimate_transform_Ls(f1, f2, maps[i] if i is not None else None, maps[j] if j is not None else None)
        tol = 0.2

    # adjacency pairs: (1->2), (2->3)
    pairs = [(0, 1), (1, 2)]
    est = {}
    # First estimate 2->3 (less ambiguous for L)
    i, j = pairs[1]
    if dtype == "circle":
        t, th, matches = estimator(feats[i], feats[j])
    else:
        t, th, matches = estimator(feats[i], feats[j], i=i, j=j)
    est[(i, j)] = (t, th, matches)
    print(f"  Estimated {i+1}-> {j+1}: theta={th:.2f} deg, t=({t[0]:.3f}, {t[1]:.3f}), matches={len(matches)}")

    # Now estimate 1->2; for L, disambiguate by consistency with 1->3 composed via 2->3
    i, j = pairs[0]
    if dtype == "circle":
        t, th, matches = estimator(feats[i], feats[j])
        est[(i, j)] = (t, th, matches)
    else:
        # generate candidates from angle-consistent pairs
        cand = []
        for a, s in enumerate(feats[i]):
            for b, d in enumerate(feats[j]):
                if abs(s.ldeg - d.ldeg) <= 5.0:
                    for t0, th0 in estimate_transform_single_L_pair_candidates(s, d):
                        cand.append((t0, th0, (a, b)))
        # score each by how well 1->2 then 2->3 aligns map1 with map3 locally
        t23, th23, _ = est[(1, 2)]
        best = None
        for t12, th12, pair in cand:
            # compose 1->3 via candidate
            t13_c, th13_c = compose_transforms(t12, th12, t23, th23)
            # evaluate alignment of map1 to map3 around the chosen centers
            from map_composer.matching import rotation_matrix_deg
            R = rotation_matrix_deg(th13_c)
            src = maps[0]
            dst = maps[2]
            # focus around corresponding centers by transforming s.lc and using NN radius
            s = feats[0][pair[0]]
            c_src = np.array(s.lc)
            src_local = src[np.linalg.norm(src - c_src, axis=1) <= 1.5]
            mapped = (R @ src_local.T).T + t13_c
            dists = []
            for p in mapped:
                d = np.linalg.norm(dst - p, axis=1)
                dists.append(float(np.min(d)))
            d = np.array(dists)
            k = max(1, len(d) // 2)
            score = float(np.sort(d)[:k].mean())
            if best is None or score < best[0]:
                best = (score, t12, th12, [pair])
        if best is None:
            t, th, matches = estimator(feats[i], feats[j], i=i, j=j)
        else:
            t, th = best[1], best[2]
            matches = best[3]
        est[(i, j)] = (t, th, matches)
    print(f"  Estimated {i+1}-> {j+1}: theta={th:.2f} deg, t=({t[0]:.3f}, {t[1]:.3f}), matches={len(matches)}")

    # Ground truth from prompt (1->2), (1->3)
    t12 = np.array([2.0, 4.0])
    th12 = 75.0
    t13 = np.array([-1.0, 0.0])
    th13 = -20.0
    # derive 2->3 = (2->1) o (1->3)
    t21, th21 = invert_transform(t12, th12)
    t23, th23 = compose_transforms(t21, th21, t13, th13)

    # Compare
    def angle_err(a, b):
        x = (a - b + 180.0) % 360.0 - 180.0
        return abs(x)

    t_e, th_e, _ = est[(0, 1)]
    print(
        f"  GT 1->2: theta={th12:.2f}, t={tuple(t12)} | Est err: dtheta={angle_err(th_e, th12):.2f}, dt=({(t_e-t12)[0]:.3f}, {(t_e-t12)[1]:.3f})"
    )
    t_e, th_e, _ = est[(1, 2)]
    print(
        f"  GT 2->3: theta={th23:.2f}, t=({t23[0]:.3f}, {t23[1]:.3f}) | Est err: dtheta={angle_err(th_e, th23):.2f}, dt=({(t_e-t23)[0]:.3f}, {(t_e-t23)[1]:.3f})"
    )

    # Plot overlays for each adjacency pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        Xs = maps[i]
        Xd = maps[j]
        t, th, _ = est[(i, j)]
        Xs_mapped = transform_points(Xs, t, th)
        ax.scatter(Xd[:, 0], Xd[:, 1], s=5, c="tab:blue", alpha=0.7, label=f"map{j+1}")
        ax.scatter(Xs_mapped[:, 0], Xs_mapped[:, 1], s=5, c="tab:orange", alpha=0.7, label=f"map{i+1}->map{j+1}")
        ax.set_title(f"{folder} map{i+1}->map{j+1}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path("out") / f"align_{base.name}_pairs.png"
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Compose to reference map1 using adjacency and estimated pairwise transforms
    edge_tf = {}
    for (i, j), (t, th, _) in est.items():
        edge_tf[(i, j)] = (t, th)
    to_ref = compute_to_ref(adj, edge_tf, root=0)
    # Compare with GT inverses
    t21_gt, th21_gt = invert_transform(t12, th12)
    t31_gt, th31_gt = invert_transform(t13, th13)
    def aerr(a,b):
        return abs((a-b+180)%360-180)
    print(f"  To map1 (BFS) t/theta:")
    for idx, (t, th) in enumerate(to_ref):
        print(f"    map{idx+1}->map1: theta={th:.2f}, t=({t[0]:.3f}, {t[1]:.3f})")
    print(
        f"  Check map2->map1 vs GT: dtheta={aerr(to_ref[1][1], th21_gt):.2f}, dt=({(to_ref[1][0]-t21_gt)[0]:.3f}, {(to_ref[1][0]-t21_gt)[1]:.3f})"
    )
    print(
        f"  Check map3->map1 vs GT: dtheta={aerr(to_ref[2][1], th31_gt):.2f}, dt=({(to_ref[2][0]-t31_gt)[0]:.3f}, {(to_ref[2][0]-t31_gt)[1]:.3f})"
    )

    # Render all maps in reference coords
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, X in enumerate(maps):
        t, th = to_ref[i]
        Xm = transform_points(X, t, th)
        ax.scatter(Xm[:, 0], Xm[:, 1], s=5, alpha=0.7, c=colors[i], label=f"map{i+1} in map1")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out2 = Path("out") / f"align_{base.name}_to_map1.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)


def main():
    estimate_for_folder("data/circle")
    estimate_for_folder("data/L")
    print("Saved overlays under out/align_circle_pairs.png and out/align_L_pairs.png")


if __name__ == "__main__":
    main()
