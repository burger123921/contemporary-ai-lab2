"""
experiments.py —— 报告中所有对照实验的可复现脚本

运行：python experiments.py            （打印表格并写出 results.json）
数据口径：A* 用 SearchStats.expanded；博弈树用 GameStats.leaves / cutoffs。
"""
from __future__ import annotations

import json
import random
import time
from typing import Dict

from astar import SearchStats
import q1_maze as q1
import q2_terrain as q2
import q3_alphabeta as q3
import q4_treasure as q4

R: Dict[str, object] = {}


def timed(fn, repeat: int = 1):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    return out, (time.perf_counter() - t0) * 1000.0 / repeat


def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- 题目一
def exp_q1():
    banner("Q1  迷宫寻路：h=0 / 曼哈顿 / 加权 A*")
    rows = {}
    for h in ("zero", "manhattan", "weighted"):
        def run(h=h):
            st_local = SearchStats()
            p = q1.solve_maze(q1.START1, q1.END1, q1.GRID1, h, st_local)
            return p, st_local

        (path, st), ms = timed(run, 200)
        rows[h] = {"expanded": st.expanded, "generated": st.generated,
                   "pushes": st.pushes, "steps": len(path) - 1,
                   "path": path, "ms_per_run": round(ms, 4)}
        print(f"  h={h:10s} expanded={st.expanded:3d} steps={len(path)-1} path={path}")
    R["q1_heuristics"] = rows

    # 自设场景：20x20 随机障碍迷宫，放大启发函数带来的差异
    for seed in range(100):
        n = 20
        rng = random.Random(seed)
        g = [["X" if rng.random() < 0.25 else "." for _ in range(n)] for _ in range(n)]
        g[0][0], g[n - 1][n - 1] = "S", "E"
        if q1.solve_maze((0, 0), (n - 1, n - 1), g, "zero"):
            break
    big = {}
    for h in ("zero", "manhattan", "weighted"):
        st = SearchStats()
        path = q1.solve_maze((0, 0), (n - 1, n - 1), g, h, st)
        big[h] = {"expanded": st.expanded, "steps": len(path) - 1 if path else -1}
        print(f"  20x20 h={h:10s} expanded={st.expanded:4d} steps={big[h]['steps']}")
    R["q1_large_maze"] = {"seed": seed, **big}


# ---------------------------------------------------------------- 题目二
def exp_q2():
    banner("Q2  加权地形：启发函数可采纳性")
    rows = {}
    for h in ("zero", "manhattan", "adjusted", "weighted"):
        st = SearchStats()
        cost, path, _ = q2.solve_terrain(q2.START2, q2.END2, q2.GRID2, h, st)
        rows[h] = {"cost": cost, "steps": len(path) - 1, "expanded": st.expanded,
                   "generated": st.generated, "path": path}
        print(f"  h={h:10s} cost={cost} expanded={st.expanded:2d} steps={len(path)-1}")
    R["q2_heuristics"] = rows

    # 代价累计过程（报告用表）
    accum, total = [], 0
    path = rows["adjusted"]["path"]
    for (r, c) in path[1:]:
        v = q2.GRID2[r][c]
        w = 0 if v == "E" else int(v)
        total += w
        accum.append({"cell": [r, c], "terrain": v, "step_cost": w, "g": total})
    R["q2_cost_accumulation"] = accum
    print("  代价累计:", " -> ".join(f"{a['step_cost']}(g={a['g']})" for a in accum))

    # 可采纳性硬证据：对每个格 n 用 Dijkstra 求真实剩余代价 h*(n)，
    # 与 h(n) 逐点比较；h(n) > h*(n) 即违反可采纳性定义。
    hm = q2.make_heuristic("manhattan", q2.END2, q2.GRID2)
    ha = q2.make_heuristic("adjusted", q2.END2, q2.GRID2)
    viol_m, viol_a, cells = [], [], 0
    for r in range(len(q2.GRID2)):
        for c in range(len(q2.GRID2[0])):
            if q2.GRID2[r][c] == "X":
                continue
            true_cost, _, _ = q2.solve_terrain((r, c), q2.END2, q2.GRID2, "zero")
            if true_cost == -1:
                continue
            cells += 1
            n = (r, c)
            if hm(n) > true_cost + 1e-9:
                viol_m.append({"cell": [r, c], "h": hm(n), "h_star": true_cost})
            if ha(n) > true_cost + 1e-9:
                viol_a.append({"cell": [r, c], "h": ha(n), "h_star": true_cost})
    R["q2_admissibility"] = {
        "checked_cells": cells,
        "manhattan_violations": len(viol_m),
        "adjusted_violations": len(viol_a),
        "examples": viol_m[:5],
    }
    print(f"  可采纳性核验：{cells} 个可通行格中，h=Manhattan 有 "
          f"{len(viol_m)} 处 h(n)>h*(n)，修正启发有 {len(viol_a)} 处")
    print("  违规示例:", viol_m[:4])


# ---------------------------------------------------------------- 题目三
def exp_q3():
    banner("Q3  Alpha-Beta：访问顺序与剪枝开销")
    tree = q3.build_sample_tree()
    base = {}
    for algo in ("minimax", "alpha_beta"):
        st = q3.GameStats()
        fn = q3.minimax if algo == "minimax" else q3.alpha_beta
        v = fn(tree, stats=st)
        base[algo] = {"value": v, **st.as_dict()}
        print(f"  给定树 {algo:10s} value={v} stats={st.as_dict()}")
    for order in ("given", "ascending", "descending", "ideal", "random"):
        st = q3.GameStats()
        v = q3.alpha_beta(tree, stats=st, order=order)
        base[f"ab_{order}"] = {"value": v, **st.as_dict()}
        print(f"  给定树 ab/{order:10s} value={v} stats={st.as_dict()}")
    memo = q3.GameStats()
    v = q3.minimax_memo(tree, stats=memo)
    base["minimax_memo_shared_dag"] = {"value": v, **memo.as_dict()}
    print(f"  给定树 记忆化Minimax value={v} stats={memo.as_dict()}")
    R["q3_sample_tree"] = base

    deep = {}
    d, b = 4, 4                       # 4 层、4 分支，共 256 个叶子
    t = q3.random_tree(d, b, seed=7)
    for algo in ("minimax", "alpha_beta"):
        st = q3.GameStats()
        fn = q3.minimax if algo == "minimax" else q3.alpha_beta
        v = fn(t, stats=st)
        deep[algo] = {"value": v, **st.as_dict()}
    for order in ("given", "ascending", "descending", "ideal", "random"):
        st = q3.GameStats()
        v = q3.alpha_beta(t, stats=st, order=order)
        deep[f"ab_{order}"] = {"value": v, **st.as_dict()}
    for k, row in deep.items():
        print(f"  深树 {k:18s} leaves={row['leaves']:4d} cutoffs={row['cutoffs']:3d} "
              f"visited={row['visited']:4d} value={row['value']}")
    R["q3_deep_tree"] = {"depth": d, "branching": b, "n_leaves": b ** d, **deep}


# ---------------------------------------------------------------- 题目四
def exp_q4():
    banner("Q4  综合题：距离矩阵、深度、算法与排序")
    D, st_m = q4.build_distance_matrix("manhattan")
    _, st_z = q4.build_distance_matrix("zero")
    print("  6x6 距离矩阵:")
    print("      " + " ".join(f"{n:>4}" for n in q4.NAMES))
    for a in q4.NAMES:
        print(f"  {a:>4}  " + " ".join(f"{D[a][b]:>4}" for b in q4.NAMES))
    print(f"  预计算开销: A*(曼哈顿) expanded={st_m.expanded}, "
          f"Dijkstra(h=0) expanded={st_z.expanded}")
    R["q4_distance_matrix"] = D
    R["q4_distance_stats"] = {"astar": st_m.as_dict(), "dijkstra": st_z.as_dict()}

    gains = {p: {t: q4.gain(p, t, D) for t in q4.TREASURES} for p in q4.NAMES}
    R["q4_gain_table"] = gains
    print("  收益表 Gain(p,T)=V-d:",
          {p: [gains[p][t] for t in q4.TREASURES] for p in q4.NAMES})

    main = {}
    for H in (2, 4):
        for algo in ("minimax", "alphabeta"):
            (res, stats), ms = timed(lambda: q4.solve(H, algo), 20)
            key = f"H{H}_{algo}"
            main[key] = {**res, "stats": stats, "ms_per_run": round(ms, 3)}
            print(f"  H={H} {algo:9s} -> {res['best_target']} {res['best_value']} "
                  f"root={res['root_values']} stats={stats}")
    R["q4_main"] = main

    orders = {}
    for order in ("given", "gain_desc", "gain_asc", "value_desc", "value_asc", "random"):
        res, stats = q4.solve(4, "alphabeta", order)
        orders[order] = {"best_target": res["best_target"],
                         "best_value": res["best_value"], **stats}
        print(f"  H=4 ab/order={order:11s} leaves={stats['leaves']:3d} "
              f"cutoffs={stats['cutoffs']:2d} nodes={stats['nodes']:3d} "
              f"-> {res['best_target']} {res['best_value']}")
    R["q4_order_effect"] = orders

    cache = {}
    for flag in (False, True):
        res, stats = q4.solve(4, "minimax", "given", use_cache=flag)
        cache["with_cache" if flag else "no_cache"] = {"best_target": res["best_target"],
                                                       "best_value": res["best_value"], **stats}
        print(f"  H=4 minimax cache={flag}: {stats}")
    R["q4_cache"] = cache

    # 拓展场景 1：把高代价格放大 2 倍（地形更崎岖）
    scaled = [["X" if v == "X" else (str(int(v) * 2) if isinstance(v, str) and v.isdigit() else v)
               for v in row] for row in q4.GRID]
    ext = {}
    for H in (2, 4):
        res, stats = q4.solve(H, "alphabeta", "given", grid=scaled)
        ext[f"H{H}_cost_x2"] = {"best_target": res["best_target"], "best_value": res["best_value"]}
    D2, _ = q4.build_distance_matrix("manhattan", scaled)
    ext["scaled_dist_A_T"] = {t: D2["A"][t] for t in q4.TREASURES}
    R["q4_extension_cost"] = ext
    print("  拓展(高代价x2):", ext)

    # 拓展场景 2：提高 T1 价值，观察深度对首步决策的影响
    orig = dict(q4.VALUE)
    q4.VALUE["T1"] = 20
    ext2 = {}
    for H in (2, 4):
        res, _ = q4.solve(H, "alphabeta", "given")
        ext2[f"H{H}_T1_value20"] = {"best_target": res["best_target"],
                                    "best_value": res["best_value"]}
    q4.VALUE.clear()
    q4.VALUE.update(orig)
    R["q4_extension_value"] = ext2
    print("  拓展(T1价值=20):", ext2)


def main():
    exp_q1()
    exp_q2()
    exp_q3()
    exp_q4()
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    banner("全部对照实验完成，结构化结果已写入 results.json")


if __name__ == "__main__":
    main()
