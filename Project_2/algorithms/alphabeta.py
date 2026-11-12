"""Alpha-beta 剪枝，可选缓存只保存完整搜索得到的精确值。"""
from math import inf
from time import perf_counter
from .minimax import GameResult


def alphabeta(state, depth, maximizing, actions, transition, evaluate,
              terminal, order=None, cache_key=None, record_trace=False):
    result, cache = GameResult(), {}
    started = perf_counter()

    def visit(s, remaining, is_max, alpha, beta, path):
        result.visited += 1
        key = (cache_key(s), remaining, is_max) if cache_key else None
        if key is not None and key in cache:
            result.cache_hits += 1
            return *cache[key], True
        if remaining == 0 or terminal(s):
            result.evaluated += 1
            value = evaluate(s)
            if record_trace:
                result.trace.append({"path": path, "value": value, "kind": "leaf"})
            if key is not None:
                cache[key] = (value, None)
            return value, None, True
        moves = list(actions(s))
        if order:
            moves = list(order(s, moves, is_max))
        value, best, exact = (-inf if is_max else inf), None, True
        for i, move in enumerate(moves):
            child, _, child_exact = visit(transition(s, move), remaining - 1,
                                          not is_max, alpha, beta, path + (move,))
            exact = exact and child_exact
            if (is_max and child > value) or (not is_max and child < value):
                value, best = child, move
            if is_max:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta and i + 1 < len(moves):
                result.cutoffs += 1
                result.skipped_children += len(moves) - i - 1
                exact = False
                if record_trace:
                    result.trace.append({"path": path, "kind": "cutoff",
                                         "alpha": alpha, "beta": beta,
                                         "skipped": moves[i + 1:]})
                break
        if not moves:
            raise ValueError("Nonterminal state has no legal action")
        # 只有所有后代都完整搜索时才缓存，避免将剪枝得到的界限当成精确值。
        if key is not None and exact:
            cache[key] = (value, best)
        if record_trace:
            result.trace.append({"path": path, "value": value, "kind": "return"})
        return value, best, exact

    result.value, result.best_action, _ = visit(state, depth, maximizing, -inf, inf, ())
    result.cache_entries = len(cache)
    result.elapsed_s = perf_counter() - started
    return result
