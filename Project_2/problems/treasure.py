from dataclasses import dataclass
from math import isfinite
from algorithms.minimax import minimax
from algorithms.alphabeta import alphabeta
from .terrain import solve_terrain

TREASURE_MAP = [["A", 1, 1, "X", 2, 1, "T1"],
                [1, "X", 6, "X", 1, "X", 1],
                [1, 1, "T2", 1, 1, 1, 1],
                ["X", 1, "X", 8, "X", 1, "X"],
                [1, 1, 1, 1, "T3", 1, 1],
                [1, "X", 9, "X", 1, "X", 1],
                ["T4", 1, 1, "X", 1, 1, "B"]]
LABELS = ("A", "B", "T1", "T2", "T3", "T4")
POSITIONS = ((0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0))
VALUES = (9, 6, 7, 7)


@dataclass(frozen=True)
class State:
    a: int = 0
    b: int = 1
    remaining: int = 15
    score_a: int = 0
    score_b: int = 0
    player: int = 0


def distance_matrix(grid=TREASURE_MAP, positions=POSITIONS):
    """逐向计算所有位置对的距离，包含自身距离，不预设矩阵对称。"""
    return [[solve_terrain(p, q, grid, free_end=False).cost for q in positions]
            for p in positions]


def solve_treasure(distances, depth=4, method="alphabeta", ordering="natural",
                   cache=False, values=VALUES, initial=None):
    if depth < 0 or method not in ("minimax", "alphabeta"):
        raise ValueError("Invalid depth or algorithm")
    if ordering not in ("natural", "reverse", "gain"):
        raise ValueError("Unknown ordering")
    initial = initial or State(remaining=(1 << len(values)) - 1)

    def actions(s):
        p = s.a if s.player == 0 else s.b
        return [i for i in range(len(values))
                if s.remaining & (1 << i) and isfinite(distances[p][i + 2])]

    def transition(s, i):
        p = s.a if s.player == 0 else s.b
        gain = values[i] - distances[p][i + 2]
        return State(i + 2 if s.player == 0 else s.a,
                     i + 2 if s.player == 1 else s.b,
                     s.remaining ^ (1 << i),
                     s.score_a + (gain if s.player == 0 else 0),
                     s.score_b + (gain if s.player == 1 else 0), 1 - s.player)

    def order(s, moves, is_max):
        if ordering == "reverse":
            return moves[::-1]
        if ordering == "gain":
            p = s.a if s.player == 0 else s.b
            # 双方都按自身即时净收益降序访问，相同时按宝藏编号排序。
            return sorted(moves, key=lambda i: (-(values[i] - distances[p][i + 2]), i))
        return moves

    kwargs = dict(order=order)
    if method == "alphabeta":
        # 缓存的是累计收益差 U，键中也要保留双方累计收益。
        kwargs["cache_key"] = (lambda s: s) if cache else None
    search = minimax if method == "minimax" else alphabeta
    return search(initial, depth, initial.player == 0, actions, transition,
                  lambda s: s.score_a - s.score_b,
                  lambda s: s.remaining == 0, **kwargs)
