"""运行固定输入的对照实验，保存原始记录和 CSV 结果表。"""
import csv
import json
import platform
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import matplotlib
import numpy
from problems.maze import MAZE, solve_maze
from problems.terrain import TERRAIN, solve_terrain
from problems.game_tree import sample_tree, solve_tree
from problems.treasure import distance_matrix, solve_treasure, LABELS
from .scenarios import grid_scenario, random_tree, larger_treasure

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 7


def measure(fn):
    fn()  # 预热一次，不计入测量。
    samples = []
    for _ in range(REPEATS):
        t = perf_counter()
        result = fn()
        samples.append(perf_counter()-t)
    row = {key: value for key, value in asdict(result).items()
           if key not in ("path", "expansion_order", "trace", "elapsed_s")}
    row["time_median_ms"] = statistics.median(samples)*1000
    row["time_samples_ms"] = [x*1000 for x in samples]
    return result, row


def save_table(name, rows):
    with (ROOT/"results/tables"/f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        keys = [k for k in rows[0] if k != "time_samples_ms"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    for folder in ("raw", "tables", "figures"):
        (ROOT/"results"/folder).mkdir(parents=True, exist_ok=True)
    output = {}
    d_started = perf_counter()
    distances = distance_matrix()
    output["distance_precompute_ms"] = (perf_counter()-d_started)*1000
    output["distance_matrix"] = distances
    with (ROOT/"results/tables/distance_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["from/to", *LABELS])
        writer.writerows([[name, *row] for name, row in zip(LABELS, distances)])

    search_rows = []
    sample_paths = {}
    for problem, grid, goal, solver, heuristics in (
        (1, MAZE, (3, 4), solve_maze, ("zero", "manhattan")),
        (2, TERRAIN, (4, 4), solve_terrain, ("zero", "admissible"))):
        for heuristic in heuristics:
            result, metrics = measure(lambda: solver((0, 0), goal, grid, heuristic))
            search_rows.append(dict(problem=problem, heuristic=heuristic, **metrics))
            sample_paths[f"p{problem}_{heuristic}"] = dict(path=result.path, expanded=result.expansion_order)
    output["sample_search"] = search_rows
    output["sample_paths"] = sample_paths
    save_table("sample_search", search_rows)

    tree_rows = []
    trace = {}
    for scene, tree in (("sample", sample_tree()), ("ternary_depth6_seed42", random_tree())):
        reference = solve_tree(tree, "minimax").value
        for method, ordering in (("minimax", "left"), ("alphabeta", "left"), ("alphabeta", "reverse")):
            result, metrics = measure(lambda: solve_tree(tree, method, ordering))
            assert result.value == reference
            tree_rows.append(dict(scene=scene, method=method, ordering=ordering, **metrics))
            if scene == "sample":
                trace[f"{method}_{ordering}"] = solve_tree(tree, method, ordering, True).trace
    output["trees"], output["sample_tree_trace"] = tree_rows, trace
    save_table("tree_search", tree_rows)

    treasure_rows = []
    for h in (2, 4):
        reference = solve_treasure(distances, h, "minimax").value
        for method, ordering, cache in (("minimax", "natural", False),
                    ("alphabeta", "natural", False), ("alphabeta", "reverse", False),
                    ("alphabeta", "gain", False), ("alphabeta", "natural", True),
                    ("alphabeta", "gain", True)):
            result, metrics = measure(lambda: solve_treasure(distances, h, method, ordering, cache))
            assert result.value == reference
            treasure_rows.append(dict(H=h, method=method, ordering=ordering, cache=cache,
                                      best_target=f"T{result.best_action+1}", **metrics))
    output["treasure"] = treasure_rows
    save_table("treasure_search", treasure_rows)
    # 逐个固定首步，再搜索剩余深度，便于解释不同首步的收益差。
    from problems.treasure import State, VALUES
    root_values = []
    for h in (2, 4):
        for i in range(4):
            gain = VALUES[i]-distances[0][i+2]
            initial = State(a=i+2, remaining=15 ^ (1 << i), score_a=gain, player=1)
            value = solve_treasure(distances, h-1, "minimax", initial=initial).value
            root_values.append(dict(H=h, target=f"T{i+1}", immediate_gain=gain, minimax_value=value))
    output["root_action_values"] = root_values
    save_table("root_action_values", root_values)
    variations = {}
    for h in (2, 4):
        state, line = State(), []
        for remaining in range(h, 0, -1):
            move = solve_treasure(distances, remaining, "minimax", initial=state).best_action
            p = state.a if state.player == 0 else state.b
            gain = VALUES[move] - distances[p][move+2]
            line.append(dict(player="A" if state.player == 0 else "B",
                             target=f"T{move+1}", gain=gain))
            state = State(move+2 if state.player == 0 else state.a,
                          move+2 if state.player == 1 else state.b,
                          state.remaining ^ (1 << move),
                          state.score_a + (gain if state.player == 0 else 0),
                          state.score_b + (gain if state.player == 1 else 0), 1-state.player)
        variations[str(h)] = line
    output["principal_variations"] = variations

    grid_rows = []
    for kind in ("open", "walls", "weighted"):
        for size in (10, 30, 60):
            for seed in (42, 43, 44, 45, 46):
                grid = grid_scenario(size, seed, kind)
                solver = solve_terrain if kind == "weighted" else solve_maze
                hs = ("zero", "admissible" if kind == "weighted" else "manhattan")
                reference = None
                for heuristic in hs:
                    result, metrics = measure(lambda: solver((0, 0), (size-1, size-1), grid, heuristic))
                    if reference is None:
                        reference = result.cost
                    assert result.cost == reference
                    grid_rows.append(dict(scene=kind, size=size, seed=seed, heuristic=heuristic, **metrics))
    output["grid_scaling"] = grid_rows
    save_table("grid_scaling", grid_rows)

    scaling_rows, scaling_inputs = [], []
    for n in (4, 6, 8):
        matrix, values, points = larger_treasure(n)
        scaling_inputs.append(dict(n=n, seed=42, points=points, values=values, distances=matrix))
        reference = solve_treasure(matrix, n, "minimax", values=values).value
        for method, ordering, cache in (("minimax", "natural", False),
                ("alphabeta", "natural", False), ("alphabeta", "gain", False),
                ("alphabeta", "natural", True), ("alphabeta", "gain", True)):
            result, metrics = measure(lambda: solve_treasure(matrix, n, method, ordering, cache, values))
            assert result.value == reference
            scaling_rows.append(dict(n=n, H=n, method=method, ordering=ordering, cache=cache, **metrics))
    output["game_scaling"], output["game_scaling_inputs"] = scaling_rows, scaling_inputs
    save_table("game_scaling", scaling_rows)

    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       processor=platform.processor(), matplotlib=matplotlib.__version__,
                       numpy=numpy.__version__, repeats=REPEATS, warmups=1,
                       grid_seeds=[42,43,44,45,46], game_seed=42,
                       tie_break="FIFO for A*; first strict improvement for games")
    (ROOT/"results/raw/environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT/"results/raw/experiments.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved experiments.json, environment.json, and seven CSV tables.")


if __name__ == "__main__":
    main()
