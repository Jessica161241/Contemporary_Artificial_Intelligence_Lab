"""A* 搜索：跳过过期堆项，允许以更低代价重新展开节点。"""
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter


@dataclass
class SearchResult:
    path: list
    cost: float
    expanded: int = 0
    generated: int = 0
    frontier_peak: int = 0
    elapsed_s: float = 0.0
    expansion_order: list = field(default_factory=list)


def astar(start, goal, neighbors, heuristic):
    """neighbors(n) 返回相邻节点及进入代价，代价必须非负。

    相同 f 按入队顺序访问。展开数包含有效出堆的终点，不计过期堆项；
    生成数包含起点入堆和每次成功松弛产生的新堆项。搜索不修改输入图。
    """
    started = perf_counter()
    serial = count()
    heap = [(heuristic(start), next(serial), 0, start)]
    best_g, parent = {start: 0}, {}
    result = SearchResult([], inf, generated=1, frontier_peak=1)
    while heap:
        _, _, g, node = heappop(heap)
        if g != best_g[node]:
            continue
        result.expanded += 1
        result.expansion_order.append(node)
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            result.path, result.cost = path[::-1], g
            break
        for successor, step in neighbors(node):
            if step < 0:
                raise ValueError("A* requires nonnegative edge costs")
            candidate = g + step
            if candidate < best_g.get(successor, inf):
                best_g[successor], parent[successor] = candidate, node
                heappush(heap, (candidate + heuristic(successor), next(serial),
                                candidate, successor))
                result.generated += 1
        result.frontier_peak = max(result.frontier_peak, len(heap))
    result.elapsed_s = perf_counter() - started
    return result
