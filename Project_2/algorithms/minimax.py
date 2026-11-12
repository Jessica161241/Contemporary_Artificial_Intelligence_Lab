"""Minimax 搜索，与 Alpha-beta 使用相同的统计口径。"""
from dataclasses import dataclass, field
from math import inf
from time import perf_counter


@dataclass
class GameResult:
    value: float = 0
    best_action: object = None
    visited: int = 0
    evaluated: int = 0
    cutoffs: int = 0
    skipped_children: int = 0
    cache_hits: int = 0
    cache_entries: int = 0
    elapsed_s: float = 0.0
    trace: list = field(default_factory=list)


def minimax(state, depth, maximizing, actions, transition, evaluate,
            terminal, order=None, record_trace=False):
    """深度表示剩余行动次数；得分相同时保留先访问的行动。"""
    result = GameResult()
    started = perf_counter()

    def visit(s, remaining, is_max, path):
        result.visited += 1
        if remaining == 0 or terminal(s):
            result.evaluated += 1
            value = evaluate(s)
            if record_trace:
                result.trace.append({"path": path, "value": value, "kind": "leaf"})
            return value, None
        moves = list(actions(s))
        if order:
            moves = list(order(s, moves, is_max))
        value, best = (-inf if is_max else inf), None
        for move in moves:
            child, _ = visit(transition(s, move), remaining - 1, not is_max,
                             path + (move,))
            if (is_max and child > value) or (not is_max and child < value):
                value, best = child, move
        if not moves:
            raise ValueError("Nonterminal state has no legal action")
        if record_trace:
            result.trace.append({"path": path, "value": value, "kind": "return"})
        return value, best

    result.value, result.best_action = visit(state, depth, maximizing, ())
    result.elapsed_s = perf_counter() - started
    return result
