#!/usr/bin/env python3
import sys
import os
import math
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pcd_xy(path: str) -> Tuple[list, list]:
    x, y = [], []
    with open(path, "r") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("DATA"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                xv = float(parts[0]); yv = float(parts[1])
            except Exception:
                continue
            x.append(xv); y.append(yv)
    return x, y


def plot_overlay(dir_in: str, out_path: str, title: str):
    files = [os.path.join(dir_in, f) for f in ["map1_in1.pcd", "map2_in1.pcd", "map3_in1.pcd"]]
    labels = ["map1_in1", "map2_in1", "map3_in1"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    plt.figure(figsize=(6,6), dpi=140)
    for fp, lb, col in zip(files, labels, colors):
        if not os.path.exists(fp):
            continue
        x, y = load_pcd_xy(fp)
        plt.scatter(x, y, s=4.0, c=col, alpha=0.7, label=lb)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(loc='best', fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv: List[str]):
    if len(argv) != 3:
        print("Usage: plot_pcd_overlay.py <in_dir> <out_png>")
        return 2
    plot_overlay(argv[1], argv[2], title=os.path.basename(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

