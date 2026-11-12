"""使用独立实现检查正确性，运行方式：python test_correctness.py。"""
import random
import unittest
from heapq import heappop, heappush
from math import inf
from problems.maze import MAZE, find_path, solve_maze
from problems.terrain import TERRAIN, minimum_cost, solve_terrain
from problems.game_tree import TreeNode, sample_tree, best_score, solve_tree, from_dict
from problems.treasure import TREASURE_MAP, POSITIONS, distance_matrix, solve_treasure
from algorithms.astar import astar


def oracle(grid, start, goal, weighted=False):
    """独立的 Dijkstra 参照实现，不调用被测代码的邻居或启发函数。"""
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return inf
    heap, best = [(0, start)], {start: 0}
    while heap:
        g, p = heappop(heap)
        if g != best[p]:
            continue
        if p == goal:
            return g
        for q in ((p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)):
            r, c = q
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] == "X":
                continue
            cell = grid[r][c]
            marker = isinstance(cell, str) and (cell in ("S", "A", "B") or cell.startswith("T"))
            cost = (0 if cell == "E" else 1 if marker else int(cell)) if weighted else 1
            if g + cost < best.get(q, inf):
                best[q] = g + cost
                heappush(heap, (g + cost, q))
    return inf


def exhaustive_game(matrix, values, h, a=0, b=1, remaining=None, u=0, player=0):
    """逐个枚举行动顺序，不调用项目中的通用博弈搜索函数。"""
    remaining = tuple(range(len(values))) if remaining is None else remaining
    if h == 0 or not remaining:
        return u
    values_out = []
    for i in remaining:
        gain = values[i] - matrix[a if player == 0 else b][i + 2]
        values_out.append(exhaustive_game(
            matrix, values, h-1, i+2 if player == 0 else a,
            i+2 if player == 1 else b, tuple(j for j in remaining if j != i),
            u + (gain if player == 0 else -gain), 1-player))
    return (max if player == 0 else min)(values_out)


class CorrectnessTests(unittest.TestCase):
    def test_sample(self):
        path = find_path((0, 0), (3, 4), MAZE)
        self.assertEqual(len(path)-1, 7)
        self.assertEqual(minimum_cost((0, 0), (4, 4), TERRAIN), 7)
        self.assertEqual(best_score(sample_tree()), 5)
        d = distance_matrix()
        for i, p in enumerate(POSITIONS):
            for j, q in enumerate(POSITIONS):
                self.assertEqual(d[i][j], oracle(TREASURE_MAP, p, q, True))
        for h, expected in ((2, (1, -1)), (4, (2, 3))):
            for method in ("minimax", "alphabeta"):
                result = solve_treasure(d, h, method)
                self.assertEqual((result.best_action, result.value), expected)

    def test_boundary(self):
        self.assertEqual(find_path((0, 0), (0, 0), [["S"]]), [(0, 0)])
        self.assertEqual(minimum_cost((0, 0), (0, 0), [["E"]]), 0)
        self.assertEqual(minimum_cost((0, 0), (0, 1), [["S", "E"]]), 0)
        self.assertEqual(find_path((0, 0), (0, 2), [["S", "X", "E"]]), [])
        self.assertEqual(minimum_cost((0, 0), (0, 2), [["S", "X", "E"]]), -1)
        self.assertEqual(find_path((0, 0), (0, 0), [["X"]]), [])
        self.assertEqual(best_score(TreeNode(-9)), -9)
        with self.assertRaises(ValueError):
            find_path((0, 0), (0, 0), [])
        with self.assertRaises(ValueError):
            solve_terrain((0, 0), (0, 1), [["S", -1]])
        with self.assertRaises(ValueError):
            from_dict({"is_maximizing_player": False, "value": 1})

    def test_random_grids(self):
        rng = random.Random(2026)
        for weighted in (False, True):
            for _ in range(100):
                grid = [["X" if rng.random() < .25 else
                         rng.randrange(0, 10) if weighted else "."
                         for _ in range(7)] for _ in range(6)]
                grid[0][0], grid[-1][-1] = "S", "E"
                expected = oracle(grid, (0, 0), (5, 6), weighted)
                result = (solve_terrain if weighted else solve_maze)((0, 0), (5, 6), grid)
                self.assertEqual(result.cost, expected)
                if result.path:
                    self.assertEqual(result.path[0], (0, 0))
                    self.assertEqual(result.path[-1], (5, 6))
                    cost = 0
                    for p, q in zip(result.path, result.path[1:]):
                        self.assertEqual(abs(p[0]-q[0])+abs(p[1]-q[1]), 1)
                        self.assertNotEqual(grid[q[0]][q[1]], "X")
                        cell = grid[q[0]][q[1]]
                        cost += (0 if cell == "E" else 1 if cell == "S" else int(cell)) if weighted else 1
                    self.assertEqual(cost, expected)

    def test_free_goal_and_directed_distance(self):
        self.assertEqual(solve_terrain((0, 0), (0, 1), [[2, 7]], free_end=False).cost, 7)
        self.assertEqual(solve_terrain((0, 1), (0, 0), [[2, 7]], free_end=False).cost, 2)
        self.assertEqual(solve_terrain((0, 0), (0, 1), [["S", "E"]], heuristic="admissible").cost, 0)

    def test_positive_weighted_heuristic(self):
        rng = random.Random(123)
        for _ in range(100):
            grid = [["X" if rng.random() < .2 else rng.randrange(1, 10)
                     for _ in range(8)] for _ in range(8)]
            grid[0][0], grid[7][7] = "S", "E"
            self.assertEqual(solve_terrain((0, 0), (7, 7), grid).cost,
                             oracle(grid, (0, 0), (7, 7), True))

    def test_astar_reopens_with_inconsistent_admissible_h(self):
        graph = {"S": [("A", 3), ("B", 1)], "B": [("A", 1)], "A": [("G", 3)], "G": []}
        h = {"S": 5, "A": 0, "B": 4, "G": 0}
        result = astar("S", "G", lambda p: graph[p], lambda p: h[p])
        self.assertEqual(result.cost, 5)
        self.assertEqual(result.path, ["S", "B", "A", "G"])

    def test_random_trees(self):
        rng = random.Random(11)
        def make(depth):
            if depth == 0 or rng.random() < .15:
                return TreeNode(rng.randrange(-20, 21))
            return TreeNode(children=[make(depth-1) for _ in range(rng.randrange(1, 5))])
        def reference(n, maximizing=True):
            if not n.children:
                return n.value
            return (max if maximizing else min)(reference(c, not maximizing) for c in n.children)
        for _ in range(100):
            root = make(5)
            expected = reference(root)
            for method in ("minimax", "alphabeta"):
                for order in ("left", "reverse"):
                    self.assertEqual(solve_tree(root, method, order).value, expected)

    def test_random_games_and_cache(self):
        rng = random.Random(97)
        for _ in range(20):
            # 使用不对称距离，检查收益正负号和距离方向是否处理正确。
            n = 5
            d = [[0 if i == j else rng.randrange(1, 12) for j in range(n+2)] for i in range(n+2)]
            values = tuple(rng.randrange(1, 12) for _ in range(n))
            for h in (2, 5):
                expected = exhaustive_game(d, values, h)
                for order in ("natural", "reverse", "gain"):
                    for cache in (False, True):
                        result = solve_treasure(d, h, ordering=order, cache=cache, values=values)
                        self.assertEqual(result.value, expected)
                        i = result.best_action
                        gain = values[i] - d[0][i+2]
                        action_value = exhaustive_game(d, values, h-1, a=i+2,
                            remaining=tuple(j for j in range(n) if j != i), u=gain, player=1)
                        self.assertEqual(action_value, expected)


if __name__ == "__main__":
    unittest.main()
