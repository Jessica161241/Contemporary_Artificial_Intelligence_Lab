"""运行四题示例或读取 JSON 输入；使用 python main.py --help 查看参数。"""
import argparse
import json
from pathlib import Path
from problems.maze import MAZE, find_path
from problems.terrain import TERRAIN, minimum_cost
from problems.game_tree import sample_tree, from_dict, best_score
from problems.treasure import distance_matrix, solve_treasure, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["all", "1", "2", "3", "4"], default="all")
    parser.add_argument("--input", type=Path, help="题一至题三的 JSON 输入文件")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    if data is not None and args.problem in ("all", "4"):
        parser.error("--input requires --problem 1, 2, or 3")
    if args.problem in ("all", "1"):
        start, goal, grid = ((tuple(data["start"]), tuple(data["goal"]), data["grid"])
                             if data else ((0, 0), (3, 4), MAZE))
        print(find_path(start, goal, grid))
    if args.problem in ("all", "2"):
        start, goal, grid = ((tuple(data["start"]), tuple(data["goal"]), data["grid"])
                             if data else ((0, 0), (4, 4), TERRAIN))
        print(minimum_cost(start, goal, grid))
    if args.problem in ("all", "3"):
        print(best_score(from_dict(data) if data else sample_tree()))
    if args.problem in ("all", "4"):
        distances = distance_matrix()
        print(json.dumps({"labels": LABELS, "distance_matrix": distances}))
        for h in (2, 4):
            for method in ("minimax", "alphabeta"):
                result = solve_treasure(distances, h, method)
                print(json.dumps({"method": method, "H": h,
                                  "best_target": f"T{result.best_action + 1}",
                                  "best_value": result.value}))


if __name__ == "__main__":
    main()
