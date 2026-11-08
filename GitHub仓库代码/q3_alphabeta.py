"""
q3_alphabeta.py —— 题目三：极大极小博弈树的最佳得分

* TreeNode 同时兼容题目给出的两种输入：
  - 「构建代码示例」：只有 value / children，按层交替 MAX/MIN（根为 MAX）；
  - 「输入样例 JSON」：节点带 is_maximizing_player 字段，以该字段为准。
* minimax / alpha_beta 共用同一统计口径 GameStats，供对照实验比较搜索开销。
* order 参数控制子节点访问顺序：given（原序，从左到右）、ascending、
  descending（按子树离线真值排序，代表「理想排序」上界）、random（固定种子）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INF = float("inf")


@dataclass
class TreeNode:
    value: Optional[int] = None
    children: List["TreeNode"] = field(default_factory=list)
    is_maximizing_player: Optional[bool] = None
    label: str = ""

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class GameStats:
    visited: int = 0     # 访问的节点总数（内部节点 + 叶子）
    leaves: int = 0      # 叶子求值次数（搜索开销的主要指标）
    cutoffs: int = 0     # 触发 β<=α 剪枝的次数
    skipped: int = 0     # 因剪枝而跳过的子树根节点数
    cache_hits: int = 0  # 记忆化命中次数

    def as_dict(self) -> Dict[str, int]:
        return {
            "visited": self.visited,
            "leaves": self.leaves,
            "cutoffs": self.cutoffs,
            "skipped": self.skipped,
            "cache_hits": self.cache_hits,
        }


def from_json(data: dict) -> TreeNode:
    """解析题目第 10-11 页的输入样例 JSON（可含 tree_structure 外壳）。"""
    raw = data.get("tree_structure", data)

    def build(d: dict) -> TreeNode:
        return TreeNode(
            value=d.get("value"),
            children=[build(c) for c in d.get("children", [])],
            is_maximizing_player=d.get("is_maximizing_player"),
        )

    return build(raw)


def build_sample_tree() -> TreeNode:
    """题目第 12 页构建代码示例（max2 被两个 MIN 节点共享）。"""
    leaf1, leaf2 = TreeNode(value=3), TreeNode(value=5)
    leaf3, leaf4 = TreeNode(value=6), TreeNode(value=9)
    leaf5, leaf6 = TreeNode(value=1), TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


SAMPLE_JSON = {
    "tree_structure": {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                ],
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                    {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
                ],
            },
        ],
    }
}
EXPECTED3 = 5


def _is_max(node: TreeNode, inherited: Optional[bool], depth: int) -> bool:
    if node.is_maximizing_player is not None:
        return node.is_maximizing_player
    if inherited is not None:
        return inherited
    return depth % 2 == 0          # 根为 MAX，逐层交替


def _true_value(node: TreeNode, cur: bool, depth: int) -> float:
    """离线计算子树真值，仅用于节点排序（不计入搜索统计）。"""
    if node.is_leaf:
        return node.value
    child_vals = [_true_value(c, not cur, depth + 1) for c in node.children]
    return max(child_vals) if cur else min(child_vals)


def ordered_children(node: TreeNode, order: str, cur: bool, depth: int) -> List[TreeNode]:
    kids = list(node.children)
    if order == "given" or len(kids) < 2:
        return kids
    if order == "random":
        random.Random(20261015 + depth).shuffle(kids)
        return kids
    if order in ("ascending", "descending"):
        kids.sort(
            key=lambda c: _true_value(c, not cur, depth + 1),
            reverse=(order == "descending"),
        )
        return kids
    if order == "ideal":
        # 「理想排序」：MAX 层先访问真值大的孩子，MIN 层先访问真值小的孩子。
        # 用离线真值作上界对照，说明节点排序能把 α-β 开销压到多低。
        kids.sort(key=lambda c: _true_value(c, not cur, depth + 1), reverse=cur)
        return kids
    raise ValueError(f"unknown order: {order}")


def minimax(
    node: TreeNode,
    is_max: Optional[bool] = None,
    stats: Optional[GameStats] = None,
    order: str = "given",
    depth: int = 0,
) -> float:
    if stats is None:
        stats = GameStats()
    stats.visited += 1
    if node.is_leaf:
        stats.leaves += 1
        return node.value
    cur = _is_max(node, is_max, depth)
    vals = [
        minimax(c, not cur, stats, order, depth + 1)
        for c in ordered_children(node, order, cur, depth)
    ]
    return max(vals) if cur else min(vals)


def alpha_beta(
    node: TreeNode,
    alpha: float = -INF,
    beta: float = INF,
    is_max: Optional[bool] = None,
    stats: Optional[GameStats] = None,
    order: str = "given",
    depth: int = 0,
) -> float:
    if stats is None:
        stats = GameStats()
    stats.visited += 1
    if node.is_leaf:
        stats.leaves += 1
        return node.value
    cur = _is_max(node, is_max, depth)
    best = -INF if cur else INF
    kids = ordered_children(node, order, cur, depth)
    for i, child in enumerate(kids):
        v = alpha_beta(child, alpha, beta, not cur, stats, order, depth + 1)
        if cur:
            best = max(best, v)
            alpha = max(alpha, best)
        else:
            best = min(best, v)
            beta = min(beta, best)
        if beta <= alpha:
            stats.cutoffs += 1
            stats.skipped += len(kids) - i - 1
            break
    return best


def minimax_memo(node: TreeNode, is_max: Optional[bool] = None,
                 stats: Optional[GameStats] = None, memo: Optional[dict] = None,
                 depth: int = 0) -> float:
    """按节点对象记忆化的 Minimax：用于验证共享子树(DAG)上的缓存收益。"""
    if stats is None:
        stats = GameStats()
    if memo is None:
        memo = {}
    key = id(node)
    if key in memo:
        stats.cache_hits += 1
        return memo[key]
    stats.visited += 1
    if node.is_leaf:
        stats.leaves += 1
        value = node.value
    else:
        cur = _is_max(node, is_max, depth)
        vals = [minimax_memo(c, not cur, stats, memo, depth + 1) for c in node.children]
        value = max(vals) if cur else min(vals)
    memo[key] = value
    return value


def random_tree(depth: int, branching: int = 4, seed: int = 7) -> TreeNode:
    """自设对照实验用的完整博弈树：depth 层、每层 branching 个孩子。"""
    rng = random.Random(seed)

    def build(d: int) -> TreeNode:
        if d == 0:
            return TreeNode(value=rng.randint(-20, 20))
        return TreeNode(children=[build(d - 1) for _ in range(branching)])

    return build(depth)


if __name__ == "__main__":
    tree = build_sample_tree()
    s1, s2 = GameStats(), GameStats()
    print("Q3 minimax:", minimax(tree, stats=s1), s1.as_dict())
    print("Q3 alpha-beta:", alpha_beta(tree, stats=s2), s2.as_dict())
    print("Q3 from JSON:", alpha_beta(from_json(json.loads(json.dumps(SAMPLE_JSON)))))
