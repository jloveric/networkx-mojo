#!/usr/bin/env python3
"""
Visualize Grid Pathfinding Results
===================================
Reads  examples/grid_paths.txt  (produced by grid_pathfinding.mojo)
and renders a matplotlib figure showing the grid, obstacles, edge-weight
heatmap, and each algorithm's path in a separate subplot.

Output is saved to  examples/grid_paths.png

Usage:
    python examples/visualize_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – no window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


# ── Parse the data file ─────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "grid_paths.txt"

if not DATA_FILE.exists():
    print(f"ERROR: {DATA_FILE} not found.  Run the Mojo example first:")
    print("    mojo run -I . examples/grid_pathfinding.mojo")
    sys.exit(1)

rows = cols = 0
obstacles: set[int] = set()
source = target = 0
paths: dict[str, list[int]] = {}
edge_weights: dict[tuple[int, int], float] = {}

with open(DATA_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "GRID":
            rows, cols = int(parts[1]), int(parts[2])
        elif tag == "OBSTACLES":
            obstacles = {int(x) for x in parts[1:]}
        elif tag == "SOURCE":
            source = int(parts[1])
        elif tag == "TARGET":
            target = int(parts[1])
        elif tag == "WEIGHTS":
            for tok in parts[1:]:
                u_s, v_s, w_s = tok.split(",")
                u, v, w = int(u_s), int(v_s), float(w_s)
                edge_weights[(u, v)] = w
                edge_weights[(v, u)] = w
        elif tag.startswith("PATH_"):
            paths[tag] = [int(x) for x in parts[1:]]


def node_to_rc(nid: int) -> tuple[int, int]:
    return divmod(nid, cols)


def path_cost(node_list: list[int]) -> float:
    """Sum of edge weights along a path."""
    total = 0.0
    for a, b in zip(node_list, node_list[1:]):
        total += edge_weights.get((a, b), 1.0)
    return total


# ── Build a per-cell average weight for the heatmap ─────────────────
cell_weight = np.full((rows, cols), np.nan)
for (u, v), w in edge_weights.items():
    r, c = node_to_rc(u)
    if np.isnan(cell_weight[r, c]):
        cell_weight[r, c] = w
    else:
        cell_weight[r, c] = max(cell_weight[r, c], w)

# Fill remaining non-obstacle cells with 1.0 (default)
for r in range(rows):
    for c in range(cols):
        nid = r * cols + c
        if nid not in obstacles and np.isnan(cell_weight[r, c]):
            cell_weight[r, c] = 1.0

# ── Colour palette ──────────────────────────────────────────────────
ALGO_META = {
    "PATH_BFS":              ("BFS (min steps)",              "#2196F3"),
    "PATH_BIDIR_BFS":        ("Bidirectional BFS (min steps)", "#4CAF50"),
    "PATH_DIJKSTRA":         ("Dijkstra (min total weight)",   "#FF9800"),
    "PATH_BIDIR_DIJKSTRA":   ("Bidir-Dijkstra (min weight)",   "#E91E63"),
    "PATH_ASTAR":            ("A* Euclidean (min weight)",     "#9C27B0"),
    "PATH_ASTAR_STEPS":       ("A* Chebyshev (min steps)",      "#3F51B5"),
}

# Weight heatmap: light yellow (cheap) → light orange (expensive)
weight_cmap = mcolors.LinearSegmentedColormap.from_list(
    "weight", ["#FFFDE7", "#FFE0B2", "#FFAB91"], N=256
)

max_w = float(np.nanmax(cell_weight))

# ── Build figure ────────────────────────────────────────────────────
n_paths = len(paths)
fig, axes = plt.subplots(1, n_paths, figsize=(4.8 * n_paths, 6.0),
                         constrained_layout=True)
if n_paths == 1:
    axes = [axes]

for ax, (key, node_list) in zip(axes, paths.items()):
    title, colour = ALGO_META.get(key, (key, "#000000"))
    cost = path_cost(node_list)

    # ── Draw weight heatmap ──────────────────────────────────────
    display = np.copy(cell_weight)
    ax.imshow(
        display,
        origin="upper",
        extent=(-0.5, cols - 0.5, rows - 0.5, -0.5),
        cmap=weight_cmap,
        vmin=1.0,
        vmax=max_w,
        alpha=0.7,
    )

    # ── Obstacles → dark grey rectangles ─────────────────────────
    for obs in obstacles:
        r, c = node_to_rc(obs)
        rect = mpatches.FancyBboxPatch(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor="#37474F", edgecolor="#263238", linewidth=0.6,
        )
        ax.add_patch(rect)

    # ── Grid lines ───────────────────────────────────────────────
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#BDBDBD", linewidth=0.3)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#BDBDBD", linewidth=0.3)

    # ── Draw path ────────────────────────────────────────────────
    path_rc = [node_to_rc(n) for n in node_list]
    pr = [r for r, c in path_rc]
    pc = [c for r, c in path_rc]
    ax.plot(pc, pr, color=colour, linewidth=2.8, zorder=3, alpha=0.9,
            solid_capstyle="round", solid_joinstyle="round")
    # Thin white outline for contrast
    ax.plot(pc, pr, color="white", linewidth=4.5, zorder=2, alpha=0.5,
            solid_capstyle="round", solid_joinstyle="round")

    # ── Source and target markers ────────────────────────────────
    sr, sc = node_to_rc(source)
    tr, tc = node_to_rc(target)
    ax.plot(sc, sr, marker="o", markersize=11, color="#00C853",
            markeredgecolor="black", markeredgewidth=1.3, zorder=5)
    ax.plot(tc, tr, marker="*", markersize=15, color="#FF1744",
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    ax.set_title(
        f"{title}\n{len(node_list)-1} steps · weight {cost:.2f}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# ── Legend ───────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#37474F", label="Obstacle"),
    mpatches.Patch(color="#FFFDE7", ec="#ccc", label="Low weight (1)"),
    mpatches.Patch(color="#FFAB91", ec="#ccc", label=f"High weight ({max_w:.2f})"),
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="#00C853",
               markeredgecolor="black", markersize=10, label="Source (0,0)"),
    plt.Line2D([], [], marker="*", color="w", markerfacecolor="#FF1744",
               markeredgecolor="black", markersize=14,
               label=f"Target ({rows-1},{cols-1})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, frameon=True, fancybox=True)

fig.suptitle("Grid Pathfinding — networkx-mojo", fontsize=14,
             fontweight="bold", y=1.01)

OUT_FILE = Path(__file__).parent / "grid_paths.png"
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Figure saved to {OUT_FILE}")
