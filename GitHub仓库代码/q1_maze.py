"""
q1_maze.py —— 题目一：迷宫寻路（单位代价 A*）

输入：起点 (row, col)、终点 (row, col)、List[List[str]] 地图（'.' 可通行，'X' 障碍）
输出：List[Tuple[int, int]] 最短路径；无解返回 []
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from astar import SearchStats, astar, grid_neighbors, manhattan

GRID1 = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
START1, END1 = (0, 0), (3, 4)
EXPECTED1 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]


def make_heuristic(name: str, end: Tuple[int, int]):
    """提供三种启发函数，用于对照实验。"""
    if name == "zero":          # h=0，退化为 Dijkstra（一致且可采纳）
        return lambda n: 0.0
    if name == "manhattan":     # 单位代价下可采纳且一致
        return lambda n: float(manhattan(n, end))
    if name == "weighted":      # 加权 A*：w>1 不再保证最优，用于拓展实验
        return lambda n: 2.0 * manhattan(n, end)
    raise ValueError(f"unknown heuristic: {name}")


def solve_maze(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grid: List[List[str]],
    heuristic: str = "manhattan",
    stats: Optional[SearchStats] = None,
) -> List[Tuple[int, int]]:
    def cost_of(r: int, c: int):
        return None if grid[r][c] == "X" else 1.0

    _, path, _ = astar(
        start,
        lambda n: n == end,
        grid_neighbors(grid, cost_of),
        make_heuristic(heuristic, end),
        stats=stats,
    )
    return [(int(r), int(c)) for r, c in path]


if __name__ == "__main__":
    st = SearchStats()
    print("Q1 path:", solve_maze(START1, END1, GRID1, stats=st))
    print("Q1 stats:", st.as_dict())
