"""
astar.py —— 通用 A* 搜索框架（仅依赖 Python 标准库）

设计要点
--------
* 状态为任意可哈希对象；neighbors(node) 产生 (next_node, step_cost)。
* open 表是 (f, h, seq, node) 的最小堆：f 相同时优先扩展 h 更小（更接近目标）
  的节点；seq 保证同 f、同 h 时按生成顺序 FIFO，因此结果确定、可复现。
* g_score / parent / closed 保证一致启发函数下每个状态最多扩展一次。
* SearchStats 统一记录 expanded / generated / relaxed / pushes / peak_open，
  报告中「搜索节点数」一律指 expanded（从 open 表有效取出并扩展的状态数）。
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from itertools import count
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

Node = Hashable
Neighbors = Callable[[Node], Iterable[Tuple[Node, float]]]
Heuristic = Callable[[Node], float]

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 上、下、左、右


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SearchStats:
    """A* 搜索开销的统计口径（见 README）。"""

    expanded: int = 0    # 有效出队并扩展的状态数（含目标状态）
    generated: int = 0   # 生成的后继边数
    relaxed: int = 0     # 成功改进 g 值的次数
    pushes: int = 0      # 入队次数
    peak_open: int = 0   # open 表最大长度

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def astar(
    start: Node,
    is_goal: Callable[[Node], bool],
    neighbors: Neighbors,
    heuristic: Heuristic,
    *,
    stats: Optional[SearchStats] = None,
    need_path: bool = True,
) -> Tuple[Optional[float], List[Node], SearchStats]:
    """返回 (最小代价, 路径, 统计)。无解时返回 (None, [], stats)。"""
    if stats is None:
        stats = SearchStats()
    seq = count()
    h0 = float(heuristic(start))
    g: Dict[Node, float] = {start: 0.0}
    parent: Dict[Node, Node] = {}
    closed = set()
    heap: List[Tuple[float, float, int, Node]] = [(h0, h0, next(seq), start)]
    stats.pushes += 1
    INF = float("inf")
    EPS = 1e-9

    while heap:
        f, h, _, node = heapq.heappop(heap)
        # 过期条目：该状态已有更小的 g，或已被关闭
        if node in closed or f > g.get(node, INF) + h + EPS:
            continue
        closed.add(node)
        stats.expanded += 1
        if is_goal(node):
            total = float(g[node])
            if not need_path:
                return total, [], stats
            path = [node]
            cur = node
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return total, path, stats
        for nxt, step_cost in neighbors(node):
            stats.generated += 1
            if nxt in closed:
                continue
            ng = g[node] + step_cost
            if ng < g.get(nxt, INF) - EPS:
                g[nxt] = ng
                parent[nxt] = node
                hn = float(heuristic(nxt))
                heapq.heappush(heap, (ng + hn, hn, next(seq), nxt))
                stats.pushes += 1
                stats.relaxed += 1
        stats.peak_open = max(stats.peak_open, len(heap))
    return None, [], stats


def grid_neighbors(grid, cost_of: Callable[[int, int], Optional[float]]) -> Neighbors:
    """把二维地图包装成四邻域 successors；cost_of 返回 None 表示障碍物。"""
    rows, cols = len(grid), len(grid[0])

    def neighbors(node):
        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                w = cost_of(nr, nc)
                if w is not None:
                    yield (nr, nc), float(w)

    return neighbors
