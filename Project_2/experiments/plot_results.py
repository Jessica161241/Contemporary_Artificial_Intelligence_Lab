"""根据已保存的实验记录和示例地图生成报告图表。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from problems.maze import MAZE
from problems.terrain import TERRAIN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/figures"
BLUE, ORANGE, RED = "#28658b", "#d08735", "#b84545"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180})


def save(fig, name):
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_grid(ax, grid, path, expanded, title):
    visited, route = set(map(tuple, expanded)), list(map(tuple, path))
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            face = "#344451" if cell == "X" else "#dceaf3" if (r, c) in visited else "#f7f7f4"
            ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor=face,
                                   edgecolor="white", linewidth=1.5))
            if cell != "X":
                ax.text(c, r, str(cell), ha="center", va="center", zorder=4)
    ax.plot([p[1] for p in route], [p[0] for p in route], color=ORANGE, lw=3, alpha=.85)
    ax.set(xlim=(-.5, len(grid[0])-.5), ylim=(len(grid)-.5, -.5),
           xticks=range(len(grid[0])), yticks=range(len(grid)), xlabel="column", ylabel="row", title=title)
    ax.set_aspect("equal")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT/"results/raw/experiments.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    counts = {r["heuristic"]: r["expanded"] for r in data["sample_search"] if r["problem"] == 1}
    for ax, key, title in zip(axes, ("p1_zero", "p1_manhattan"),
                              (f"h = 0 | expanded = {counts['zero']}",
                               f"Manhattan | expanded = {counts['manhattan']}")):
        p = data["sample_paths"][key]
        draw_grid(ax, MAZE, p["path"], p["expanded"], title)
    fig.suptitle("Problem 1: blue cells = expanded; orange line = shortest path")
    save(fig, "maze_search")

    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    p = data["sample_paths"]["p2_admissible"]
    draw_grid(ax, TERRAIN, p["path"], p["expanded"], "Problem 2: cost = 7; entry to E is free")
    save(fig, "weighted_path")

    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    locations = {(): (3.5, 3), (0,): (1.5, 2), (1,): (5.5, 2)}
    locations.update({(a,b): (a*4+b*2+.5, 1) for a in range(2) for b in range(2)})
    leaf_values = [3,5,6,9,6,9,1,2]
    for i, v in enumerate(leaf_values):
        locations[(i//4, (i//2)%2, i%2)] = (i, 0)
    labels = {(): "MAX\n5", (0,): "MIN\n5", (1,): "MIN\n2",
              (0,0): "MAX\n5", (0,1): "MAX\n>=6", (1,0): "MAX\n9", (1,1): "MAX\n2"}
    for i,v in enumerate(leaf_values):
        labels[(i//4, (i//2)%2, i%2)] = str(v)
    for key, (x,y) in locations.items():
        pruned = key == (0,1,1)
        if key:
            px,py = locations[key[:-1]]
            ax.plot([px,x], [py,y], color=RED if pruned else "#a8b2ba",
                    ls="--" if pruned else "-", zorder=0)
        ax.text(x,y,labels[key],ha="center",va="center",color=RED if pruned else "#203b4c",
                bbox=dict(boxstyle="round,pad=.35", fc="#fff0ee" if pruned else "#e7f0f5", ec="none"))
    ax.annotate("alpha = 6 >= beta = 5\nskip the leaf 9", xy=(2.8,.4), xytext=(3.3,1.5),
                color=RED, arrowprops=dict(arrowstyle="->",color=RED))
    ax.set(xlim=(-.5,7.5),ylim=(-.4,3.5),title="Problem 3: left-to-right alpha-beta (>=6 is a bound, not an exact subtree value)")
    ax.axis("off")
    save(fig, "tree_pruning")

    fig, axes = plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax, scene in zip(axes,("open","walls","weighted")):
        hs=("zero","admissible" if scene=="weighted" else "manhattan")
        for h,color in zip(hs,(BLUE,ORANGE)):
            means, errors=[],[]
            for size in (10,30,60):
                values=[r["expanded"] for r in data["grid_scaling"] if r["scene"]==scene and r["size"]==size and r["heuristic"]==h]
                means.append(np.mean(values)); errors.append(np.std(values,ddof=1))
            ax.errorbar((10,30,60),means,yerr=errors,label=h,marker="o",capsize=3,color=color)
        ax.set(xlabel="Grid side length",ylabel="Expanded nodes",title=scene,yscale="log")
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Grid scaling: mean +/- sample SD over five seeds; guaranteed boundary route")
    save(fig, "grid_scaling")

    fig, ax = plt.subplots(figsize=(7.5,4),layout="constrained")
    for method,ordering,cache,label,color,style in (
        ("minimax","natural",False,"Minimax", "#82929c","--"),
        ("alphabeta","natural",False,"Alpha-beta",BLUE,"-"),
        ("alphabeta","gain",False,"Alpha-beta + gain order",ORANGE,"-"),
        ("alphabeta","gain",True,"Alpha-beta + order + cache",RED,":")):
        rows=[r for r in data["game_scaling"] if r["method"]==method and r["ordering"]==ordering and r["cache"]==cache]
        ax.plot([r["n"] for r in rows],[r["visited"] for r in rows],marker="o",label=label,color=color,ls=style)
    ax.set(xlabel="Number of treasures (H = n)",ylabel="Visited calls (log scale)",yscale="log",xticks=(4,6,8),
           title="Game scaling on unit grids | fixed seed 42")
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save(fig, "game_scaling")
    print("Saved five figures to results/figures.")


if __name__ == "__main__":
    main()
