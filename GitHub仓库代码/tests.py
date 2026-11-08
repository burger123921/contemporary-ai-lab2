"""
tests.py —— 四道题目给定测试用例的自动断言（python tests.py）
"""
from __future__ import annotations

import json

from astar import SearchStats
import q1_maze as q1
import q2_terrain as q2
import q3_alphabeta as q3
import q4_treasure as q4


def test_q1():
    path = q1.solve_maze(q1.START1, q1.END1, q1.GRID1)
    assert path == q1.EXPECTED1, f"Q1 path mismatch: {path}"
    assert q1.solve_maze((0, 0), (4, 4), [["S", "X"], ["X", "E"]]) == []
    print("[PASS] Q1 迷宫寻路：路径与无解情形")


def test_q2():
    cost, path, _ = q2.solve_terrain(q2.START2, q2.END2, q2.GRID2)
    assert cost == q2.EXPECTED2, f"Q2 cost mismatch: {cost}"
    assert path == q2.EXPECTED2_PATH, f"Q2 path mismatch: {path}"
    print("[PASS] Q2 加权地形：最小代价 7，路径与题目一致")


def test_q3():
    tree = q3.build_sample_tree()
    assert q3.minimax(tree) == q3.EXPECTED3
    assert q3.alpha_beta(tree) == q3.EXPECTED3
    assert q3.alpha_beta(q3.from_json(json.loads(json.dumps(q3.SAMPLE_JSON)))) == q3.EXPECTED3
    print("[PASS] Q3 Alpha-Beta：TreeNode 与 JSON 两种输入均得 5")


def test_q4():
    D = q4.DIST
    for a in q4.NAMES:
        for b in q4.NAMES:
            assert D[a][b] == D[b][a], "距离矩阵应对称（无向网格）"
    for H in (2, 4):
        for algo in ("minimax", "alphabeta"):
            res, _ = q4.solve(H, algo)
            assert res["best_target"] == q4.EXPECTED[H]["best_target"], (H, algo, res)
            assert res["best_value"] == q4.EXPECTED[H]["best_value"], (H, algo, res)
    for order in ("given", "gain_desc", "gain_asc", "value_desc", "random"):
        res, _ = q4.solve(4, "alphabeta", order)
        assert res["best_value"] == q4.EXPECTED[4]["best_value"], (order, res)
    print("[PASS] Q4 综合题：H=2/4、两种算法、五种访问顺序结果一致")


if __name__ == "__main__":
    test_q1()
    test_q2()
    test_q3()
    test_q4()
    print("全部 4 组给定测试用例通过。")
