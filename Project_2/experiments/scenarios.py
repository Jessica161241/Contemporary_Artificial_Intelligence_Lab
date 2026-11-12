import random
from problems.game_tree import TreeNode


def grid_scenario(size, seed, kind):
    rng = random.Random(seed)
    if kind == "open":
        grid = [["."] * size for _ in range(size)]
    elif kind == "walls":
        grid = [["X" if rng.random() < .27 else "." for _ in range(size)] for _ in range(size)]
    elif kind == "weighted":
        grid = [["X" if rng.random() < .2 else rng.choice((1, 1, 1, 3, 8))
                 for _ in range(size)] for _ in range(size)]
    else:
        raise ValueError(kind)
    # 保留顶行和最右列的通路，保证地图有解；分析时需说明这一设置。
    for i in range(size):
        grid[0][i] = grid[i][-1] = 1 if kind == "weighted" else "."
    grid[0][0], grid[-1][-1] = "S", "E"
    return grid


def random_tree(seed=42, depth=6, branching=3):
    rng = random.Random(seed)
    def make(d):
        if d == 0:
            return TreeNode(rng.randrange(-100, 101))
        return TreeNode(children=[make(d-1) for _ in range(branching)])
    return make(depth)


def larger_treasure(n, seed=42):
    """在 20×20 无障碍单位网格中，曼哈顿距离就是精确距离。

    固定随机种子并抽取互异坐标，单独比较博弈搜索的规模开销。
    此处不重复运行 A*，寻路性能由网格实验另行比较。
    """
    rng = random.Random(seed)
    points = rng.sample([(r, c) for r in range(20) for c in range(20)], n+2)
    matrix = [[abs(p[0]-q[0])+abs(p[1]-q[1]) for q in points] for p in points]
    values = tuple(rng.randrange(5, 26) for _ in range(n))
    return matrix, values, points
