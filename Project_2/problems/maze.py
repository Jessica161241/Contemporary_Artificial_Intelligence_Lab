from algorithms.astar import astar

# 邻居按右、下、左、上访问，相同优先级按入队顺序处理，使路径可复现。
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MAZE = [list(row) for row in ("S..X.", ".X...", ".X.X.", "...XE", "XX...")]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_grid(grid, start, goal):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Grid must be nonempty and rectangular")
    for r, c in (start, goal):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            raise ValueError("Coordinate outside grid")


def solve_maze(start, goal, grid, heuristic="manhattan"):
    validate_grid(grid, start, goal)
    if heuristic not in ("zero", "manhattan"):
        raise ValueError("Unknown heuristic")

    def neighbors(p):
        if grid[start[0]][start[1]] == "X":
            return
        for dr, dc in DIRECTIONS:
            r, c = p[0] + dr, p[1] + dc
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X":
                yield (r, c), 1

    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        from algorithms.astar import SearchResult
        return SearchResult([], float("inf"))
    h = (lambda p: 0) if heuristic == "zero" else (lambda p: manhattan(p, goal))
    return astar(start, goal, neighbors, h)


def find_path(start, goal, grid):
    """题一接口：返回坐标元组列表，无解时返回空列表。"""
    return solve_maze(start, goal, grid).path
