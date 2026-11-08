"""
run_all.py —— 一键复现：四题给定测试用例 + 对照实验摘要
    python run_all.py
"""
from __future__ import annotations

from astar import SearchStats
import q1_maze as q1
import q2_terrain as q2
import q3_alphabeta as q3
import q4_treasure as q4
import tests


def main():
    print("=" * 72)
    print("当代人工智能实验课 实验二：搜索算法 —— 全部结果复现")
    print("=" * 72)
    st1 = SearchStats()
    print("Q1 迷宫寻路 :", q1.solve_maze(q1.START1, q1.END1, q1.GRID1, stats=st1))
    print("   A* 扩展节点数:", st1.expanded)
    cost2, path2, st2 = q2.solve_terrain(q2.START2, q2.END2, q2.GRID2)
    print("Q2 加权地形 : cost =", cost2, " 路径 =", path2, " 扩展节点数 =", st2.expanded)
    s3a, s3b = q3.GameStats(), q3.GameStats()
    print("Q3 Minimax  :", q3.minimax(q3.build_sample_tree(), stats=s3a), s3a.as_dict())
    print("Q3 AlphaBeta:", q3.alpha_beta(q3.build_sample_tree(), stats=s3b), s3b.as_dict())
    print("\nQ4 6x6 距离矩阵 (行/列 = A,B,T1,T2,T3,T4):")
    print("      " + " ".join(f"{n:>4}" for n in q4.NAMES))
    for a in q4.NAMES:
        print(f"  {a:>4}  " + " ".join(f"{q4.DIST[a][b]:>4}" for b in q4.NAMES))
    for H in (2, 4):
        for algo in ("minimax", "alphabeta"):
            res, stats = q4.solve(H, algo)
            print(f"Q4 H={H} {algo:9s}: {res['best_target']} {res['best_value']} "
                  f"root={res['root_values']} stats={stats}")
    print("\n--- 自动断言 ---")
    tests.test_q1()
    tests.test_q2()
    tests.test_q3()
    tests.test_q4()
    print("\n对照实验与拓展数据请运行: python experiments.py (生成 results.json)")


if __name__ == "__main__":
    main()
