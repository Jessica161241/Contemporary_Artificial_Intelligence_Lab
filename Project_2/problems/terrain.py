from algorithms.astar import astar, SearchResult
from .maze import DIRECTIONS, manhattan, validate_grid

TERRAIN = [["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
           [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]]


def solve_terrain(start, goal, grid, heuristic="admissible", free_end=True):
    """题二规定 S=1、E=0；宝藏问题使用 free_end=False。

    A、B、Ti 的进入代价为 1，X 表示障碍，数字代价必须为非负整数。
    起点代价不计入路径总成本。
    """
    validate_grid(grid, start, goal)
    if heuristic not in ("zero", "admissible", "raw_manhattan"):
        raise ValueError("Unknown heuristic")
    costs = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "X":
                continue
            if isinstance(cell, str) and (cell in ("S", "E", "A", "B") or cell.startswith("T")):
                cost = 0 if cell == "E" and free_end else 1
            else:
                try:
                    cost = int(cell)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid terrain cost") from exc
                if str(cost) != str(cell) or cost < 0:
                    raise ValueError("Terrain costs must be nonnegative integers")
            costs[(r, c)] = cost
    if start not in costs or goal not in costs:
        return SearchResult([], float("inf"))
    # 计算最低代价时只排除目标格；若还有其他免费格，下界退化为零。
    free_goal = costs[goal] == 0
    c_min = min((v for p, v in costs.items() if p != goal), default=0)
    if not free_goal:
        c_min = min(c_min, costs[goal])

    def h(p):
        distance = manhattan(p, goal)
        if heuristic == "zero":
            return 0
        if heuristic == "raw_manhattan":
            return distance  # 仅用于研究原始曼哈顿距离高估的情况。
        return c_min * max(distance - int(free_goal), 0)

    def neighbors(p):
        for dr, dc in DIRECTIONS:
            q = p[0] + dr, p[1] + dc
            if q in costs:
                yield q, costs[q]

    return astar(start, goal, neighbors, h)


def minimum_cost(start, goal, grid):
    """题二接口：返回整数最小代价，无解时返回 -1。"""
    result = solve_terrain(start, goal, grid)
    return int(result.cost) if result.path else -1
