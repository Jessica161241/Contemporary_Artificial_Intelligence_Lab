from algorithms.minimax import minimax
from algorithms.alphabeta import alphabeta


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def sample_tree():
    def branch(a, b):
        return TreeNode(children=[TreeNode(a), TreeNode(b)])
    # 两组相同叶子分别建树，避免共享节点影响访问过程的统计。
    return TreeNode(children=[TreeNode(children=[branch(3, 5), branch(6, 9)]),
                              TreeNode(children=[branch(6, 9), branch(1, 2)])])


def from_dict(data, maximizing=True):
    if "tree_structure" in data:
        data = data["tree_structure"]
    if "is_maximizing_player" in data and data["is_maximizing_player"] != maximizing:
        raise ValueError("Player flag must alternate, starting with MAX")
    children = data.get("children", [])
    return TreeNode(data.get("value"), [from_dict(c, not maximizing) for c in children])


def solve_tree(root, method="alphabeta", ordering="left", record_trace=False):
    if method not in ("minimax", "alphabeta") or ordering not in ("left", "reverse"):
        raise ValueError("Unknown method or ordering")

    def evaluate(node):
        if node.value is None:
            raise ValueError("Leaf must have a value")
        return node.value

    search = minimax if method == "minimax" else alphabeta
    order = (lambda s, moves, is_max: reversed(moves)) if ordering == "reverse" else None
    return search(root, float("inf"), True,
                  lambda n: range(len(n.children)), lambda n, i: n.children[i],
                  evaluate, lambda n: not n.children, order=order,
                  record_trace=record_trace)


def best_score(root):
    """题三接口：从左到右访问，返回 MAX 根节点的最优得分。"""
    return solve_tree(root).value
