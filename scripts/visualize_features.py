import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from map_composer import (
    load_maps,
    load_adjacency,
    infer_dataset_type,
    extract_circle_features,
    extract_L_features,
)


def plot_circle_dataset(folder: str, out_path: str) -> None:
    maps = load_maps([str(Path(folder) / f"map{i}.csv") for i in (1, 2, 3)])
    feats = extract_circle_features(maps)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for idx, (ax, pts, f_list) in enumerate(zip(axes[0], maps, feats), start=1):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, c="tab:blue", alpha=0.7, label="points")
        for f in f_list:
            cx, cy = f.center
            ax.scatter([cx], [cy], marker="x", s=60, c="tab:red", label="center" if idx == 1 else None)
            # optional: draw circle outline
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(cx + f.radius * np.cos(theta), cy + f.radius * np.sin(theta), c="tab:orange", lw=0.8)
        ax.set_title(f"circle map{idx}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.5)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_L_dataset(folder: str, out_path: str) -> None:
    maps = load_maps([str(Path(folder) / f"map{i}.csv") for i in (1, 2, 3)])
    feats_per_map = extract_L_features(maps)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for idx, (ax, pts, feats) in enumerate(zip(axes[0], maps, feats_per_map), start=1):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, c="tab:blue", alpha=0.7, label="points")
        xmn, xmx = np.min(pts[:, 0]), np.max(pts[:, 0])
        ymn, ymx = np.min(pts[:, 1]), np.max(pts[:, 1])
        pad = 0.1 * max(xmx - xmn, ymx - ymn)
        xmn, xmx = xmn - pad, xmx + pad
        ymn, ymx = ymn - pad, ymx + pad
        for k, f in enumerate(feats):
            for line, color in zip([f.line1, f.line2], ["tab:orange", "tab:green"]):
                p0, d = line
                corners = np.array([[xmn, ymn], [xmn, ymx], [xmx, ymn], [xmx, ymx]])
                t_vals = ((corners - p0) @ d)
                tmin, tmax = float(np.min(t_vals)), float(np.max(t_vals))
                P1 = p0 + tmin * d
                P2 = p0 + tmax * d
                ax.plot([P1[0], P2[0]], [P1[1], P2[1]], c=color, lw=1.1, alpha=0.9)
            ax.scatter([f.lc[0]], [f.lc[1]], marker="*", s=80, c="tab:red", label="intersection" if (idx == 1 and k == 0) else None)
        ax.set_title(f"L map{idx} (k={len(feats)})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmn, xmx)
        ax.set_ylim(ymn, ymx)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # Type checks (not strictly needed for plotting, but illustrative)
    adj = load_adjacency("data/adj.csv")
    for folder in ["data/circle", "data/L"]:
        maps = [str(Path(folder) / f"map{i}.csv") for i in (1, 2, 3)]
        dtype = infer_dataset_type(maps, adj)
        print(f"{folder}: inferred type = {dtype}")

    plot_circle_dataset("data/circle", "out/circle_features.png")
    plot_L_dataset("data/L", "out/L_features.png")
    print("Saved out/circle_features.png and out/L_features.png")


if __name__ == "__main__":
    main()
