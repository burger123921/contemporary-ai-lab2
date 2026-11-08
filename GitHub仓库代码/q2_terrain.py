"""
q2_terrain.py —— 题目二：加权地形的最小移动代价

规则：进入格 B 的代价 = B 的地形权值；S 起点无进入代价；E 视为 0（题目说明）。
输出：最小总代价 int；无解返回 -1。

启发函数讨论（报告第 2 节）：
  设 w_min 为除 S/E/X 外的最小地形权值，本题 w_min=1，k=Manhattan(n,E)。
  剩余至少 k 步，其中最后一步进入 E 代价为 0，故真实剩余代价 >= (k-1)*w_min。
  * h=Manhattan(n,E) = k*w_min 高估了最后一步 -> 严格来说不可采纳；
  * h=max(0,(k-1)*w_min) 才是可采纳且一致的启发函数。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from astar import SearchStats, astar, grid_neighbors, manhattan

GRID2 = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
START2, END2 = (0, 0), (4, 4)
EXPECTED2 = 7
EXPECTED2_PATH = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]


def min_terrain_cost(grid) -> float:
    """除 S/E/X 之外的最小进入代价，用于构造可采纳启发函数。"""
    vals = []
    for row in grid:
        for v in row:
            if v in ("S", "E", "X"):
                continue
            if isinstance(v, str) and not v.isdigit():
                continue
            vals.append(float(v))
    return min(vals) if vals else 1.0


def make_heuristic(name: str, end: Tuple[int, int], grid):
    w = min_terrain_cost(grid)
    k = lambda n: manhattan(n, end)
    if name == "zero":                       # Dijkstra，可采纳（平凡下界）
        return lambda n: 0.0
    if name == "manhattan":                  # 题目提示的朴素做法，本题不可采纳
        return lambda n: w * k(n)
    if name == "adjusted":                   # 修正 E 免费后的可采纳启发
        return lambda n: max(0.0, w * (k(n) - 1))
    if name == "weighted":                   # 加权 A*（拓展实验，不保证最优）
        return lambda n: 1.5 * max(0.0, w * (k(n) - 1))
    raise ValueError(f"unknown heuristic: {name}")


def solve_terrain(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grid: List[List],
    heuristic: str = "adjusted",
    stats: Optional[SearchStats] = None,
):
    """返回 (最小代价 int 或 -1, 路径, 统计)。"""
    def cost_of(r: int, c: int):
        v = grid[r][c]
        if v == "X":
            return None
        if v in ("S", "E"):
            return 0.0          # 起点与终点 E 不产生进入代价
        return float(v)

    cost, path, st = astar(
        start,
        lambda n: n == end,
        grid_neighbors(grid, cost_of),
        make_heuristic(heuristic, end, grid),
        stats=stats,
    )
    return (-1 if cost is None else int(cost)), path, st


if __name__ == "__main__":
    print("Q2:", solve_terrain(START2, END2, GRID2))
