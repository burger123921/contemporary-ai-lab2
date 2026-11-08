"""
q4_treasure.py —— 题目四：竞争环境下的宝藏分配（A* + Minimax/Alpha-Beta 综合题）

建模
----
* 6 个关键格：A、B 与四个宝藏 T1..T4；进入代价 = 数字，A/B/T 所在格代价 1，X 不可通行。
* d(p, Ti)：从当前位置 p 到 Ti 的最小移动代价，用带曼哈顿启发的 A* 预计算，
  得到 6x6 距离矩阵；因为 w_min=1 且终点格代价为 1，曼哈顿距离可采纳且一致。
* Gain(p, Ti) = V_i - d(p, Ti)；U = Score_A - Score_B；A 最大化 U，B 最小化 U。
* 博弈状态 State(posA, posB, remaining, scoreA, scoreB, ply)，
  行动顺序 A -> B -> A -> B，深度 H 到达或宝藏取完时用当前 U 评估。
* 递归统一返回「相对当前 U 的增量」，因此置换表可以跨不同绝对收益复用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from astar import SearchStats, astar, grid_neighbors, manhattan

GRID = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]
POS: Dict[str, Tuple[int, int]] = {
    "A": (0, 0), "B": (6, 6),
    "T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
VALUE: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
NAMES: List[str] = ["A", "B", "T1", "T2", "T3", "T4"]
TREASURES: List[str] = ["T1", "T2", "T3", "T4"]
EXPECTED = {2: {"best_target": "T2", "best_value": -1},
            4: {"best_target": "T3", "best_value": 3}}
INF = float("inf")


def cell_cost(r: int, c: int, grid: Optional[List[List[str]]] = None) -> Optional[float]:
    grid = grid or GRID
    v = grid[r][c]
    if v == "X":
        return None
    if v in POS:                 # A、B、T 所在格代价为 1
        return 1.0
    return float(v)


def build_distance_matrix(
    heuristic: str = "manhattan", grid: Optional[List[List[str]]] = None
) -> Tuple[Dict[str, Dict[str, int]], SearchStats]:
    """6x6 最小代价矩阵；heuristic='zero' 时为 Dijkstra（对照实验）。
    传入 grid 可在不改变默认地图的前提下做自设场景拓展实验。"""
    grid = grid or GRID

    def hfun(p, goal):
        return manhattan(p, goal) if heuristic == "manhattan" else 0

    dist: Dict[str, Dict[str, int]] = {a: {} for a in NAMES}
    stats = SearchStats()
    for a in NAMES:
        for b in NAMES:
            if a == b:
                dist[a][b] = 0
                continue
            cost, _, _ = astar(
                POS[a], lambda n: n == POS[b],
                grid_neighbors(grid, lambda r, c, g=grid: cell_cost(r, c, g)),
                lambda n, g=POS[b]: float(hfun(n, g)),
                stats=stats, need_path=False,
            )
            if cost is None:
                raise RuntimeError(f"no path between {a} and {b}")
            dist[a][b] = int(cost)
    return dist, stats


DIST, DIST_STATS = build_distance_matrix()


def gain(p: str, t: str, D: Optional[Dict[str, Dict[str, int]]] = None) -> int:
    D = D or DIST
    return VALUE[t] - D[p][t]


@dataclass(frozen=True)
class State:
    posA: str
    posB: str
    remaining: FrozenSet[str]
    scoreA: int = 0
    scoreB: int = 0
    ply: int = 0

    @property
    def U(self) -> int:
        return self.scoreA - self.scoreB


@dataclass
class GameStats:
    nodes: int = 0        # 展开的博弈节点数
    leaves: int = 0       # 到达深度/宝藏取完时的叶评估数
    cutoffs: int = 0      # 触发剪枝次数
    skipped: int = 0      # 被剪掉的动作数
    cache_hits: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"nodes": self.nodes, "leaves": self.leaves, "cutoffs": self.cutoffs,
                "skipped": self.skipped, "cache_hits": self.cache_hits}


def ordered_actions(rem: FrozenSet[str], p: str, order: str, ply: int,
                    D: Optional[Dict[str, Dict[str, int]]] = None) -> List[str]:
    acts = sorted(rem)                      # 原序：从左到右 T1..T4
    if order == "gain_desc":               # 立即收益优先（贪心启发式排序）
        acts.sort(key=lambda t: -gain(p, t, D))
    elif order == "gain_asc":
        acts.sort(key=lambda t: gain(p, t, D))
    elif order == "value_desc":            # 高价值宝藏优先
        acts.sort(key=lambda t: -VALUE[t])
    elif order == "value_asc":
        acts.sort(key=lambda t: VALUE[t])
    elif order == "random":
        random.Random(1015 + ply).shuffle(acts)
    elif order != "given":
        raise ValueError(f"unknown order: {order}")
    return acts


def solve(
    H: int,
    algorithm: str = "minimax",
    order: str = "given",
    use_cache: bool = False,
    dist_heuristic: str = "manhattan",
    grid: Optional[List[List[str]]] = None,
):
    """返回 (结果 dict, 统计 dict)。结果含 best_target / best_value / root_values。"""
    D, _ = build_distance_matrix(dist_heuristic, grid)
    g = lambda p, t: gain(p, t, D)         # 闭包内的收益函数，避免全局状态
    st = GameStats()
    cache: Dict[tuple, float] = {}

    def terminal(posA, posB, rem, ply) -> bool:
        return ply >= H or not rem

    def minimax_rec(posA, posB, rem, ply) -> float:
        st.nodes += 1
        if terminal(posA, posB, rem, ply):
            st.leaves += 1
            return 0.0
        key = (posA, posB, rem, ply)
        if use_cache and key in cache:
            st.cache_hits += 1
            return cache[key]
        is_a = ply % 2 == 0
        p = posA if is_a else posB
        best = -INF if is_a else INF
        for t in ordered_actions(rem, p, order, ply, D):
            delta = g(p, t) * (1 if is_a else -1)
            rest = minimax_rec(t if is_a else posA, posB if is_a else t,
                               rem - {t}, ply + 1)
            v = delta + rest
            best = max(best, v) if is_a else min(best, v)
        if use_cache:
            cache[key] = best
        return best

    def ab_rec(posA, posB, rem, ply, alpha, beta) -> float:
        st.nodes += 1
        if terminal(posA, posB, rem, ply):
            st.leaves += 1
            return 0.0
        is_a = ply % 2 == 0
        p = posA if is_a else posB
        best = -INF if is_a else INF
        acts = ordered_actions(rem, p, order, ply, D)
        for i, t in enumerate(acts):
            delta = g(p, t) * (1 if is_a else -1)
            rest = ab_rec(t if is_a else posA, posB if is_a else t,
                          rem - {t}, ply + 1, alpha, beta)
            v = delta + rest
            if is_a:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            if beta <= alpha:
                st.cutoffs += 1
                st.skipped += len(acts) - i - 1
                break
        # α-β 返回的可能只是窗口界，置换表需保存 bound/flag；本实验只在
        # Minimax 上启用缓存，避免把界值误当作精确值。
        return best

    rem0: FrozenSet[str] = frozenset(TREASURES)
    root_values: Dict[str, int] = {}
    for t in ordered_actions(rem0, "A", order, 0, D):
        delta = g("A", t)
        if algorithm == "minimax":
            rest = minimax_rec(t, "B", rem0 - {t}, 1)
        elif algorithm == "alphabeta":
            rest = ab_rec(t, "B", rem0 - {t}, 1, -INF, INF)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        root_values[t] = int(delta + rest)
    best_t = max(root_values, key=lambda k: root_values[k])
    result = {
        "H": H,
        "best_target": best_t,
        "best_value": root_values[best_t],
        "root_values": dict(sorted(root_values.items())),
        "algorithm": algorithm,
        "order": order,
    }
    return result, st.as_dict()


if __name__ == "__main__":
    print("6x6 distance matrix (rows/cols = A,B,T1,T2,T3,T4):")
    print("      " + " ".join(f"{n:>4}" for n in NAMES))
    for a in NAMES:
        print(f"{a:>4}  " + " ".join(f"{DIST[a][b]:>4}" for b in NAMES))
    for H in (2, 4):
        for algo in ("minimax", "alphabeta"):
            res, s = solve(H, algo)
            print(f"H={H} {algo:9s}: {res['best_target']} {res['best_value']}  "
                  f"root={res['root_values']}  stats={s}")
