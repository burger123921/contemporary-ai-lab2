# 当代人工智能实验课 · 实验二：搜索算法

A\* 搜索、加权地形启发函数可采纳性、Minimax / Alpha-Beta 剪枝，以及
「竞争环境下的宝藏分配」综合题的完整实现与可复现对照实验。

> 课程：《当代人工智能》实验二（搜索算法）
> 语言：Python 3.9+，**仅使用标准库**，无第三方依赖。

## 快速开始

```bash
git clone <本仓库地址>
cd contemporary-ai-lab2
python tests.py          # 4 组给定测试用例断言
python run_all.py        # 一键打印四题全部结果
python experiments.py    # 运行全部对照实验，生成 results.json
```

`results.json` 是实验报告中所有表格与结论的数据来源，便于逐项核对。

## 文件结构

| 文件 | 内容 |
| --- | --- |
| `astar.py` | 通用 A\* 框架：open 最小堆 `(f, h, seq, node)`、g/parent/closed、SearchStats 统计 |
| `q1_maze.py` | 题目一 迷宫寻路，`solve_maze(start, end, grid, heuristic)` → 路径坐标列表 |
| `q2_terrain.py` | 题目二 加权地形，`solve_terrain(start, end, grid, heuristic)` → (最小代价, 路径, 统计) |
| `q3_alphabeta.py` | 题目三 TreeNode、Minimax、Alpha-Beta、四种子节点访问顺序、随机深树生成器 |
| `q4_treasure.py` | 题目四 地图/收益建模、A\* 距离矩阵、博弈状态、Minimax/Alpha-Beta、置换表、拓展场景 |
| `experiments.py` | 全部对照实验（启发函数、访问顺序、深度、缓存、自设场景与反例） |
| `tests.py` | 给定测试用例的自动断言 |
| `run_all.py` | 一键复现入口 |
| `results.json` | `python experiments.py` 生成的结构化实验数据 |

## 给定测试用例结果

| 题目 | 预期 | 本实现 |
| --- | --- | --- |
| 一 迷宫寻路 | `[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4)]` | 一致（7 步） |
| 二 加权地形 | `7` | 一致（E 进入代价 0） |
| 三 Alpha-Beta | `5` | 一致（TreeNode 与 JSON 两种输入） |
| 四 H=2 | `{"best_target":"T2","best_value":-1}` | 一致 |
| 四 H=4 | `{"best_target":"T3","best_value":3}` | 一致 |

## 关键实现约定

1. **A\* 状态**：网格坐标 `(row, col)`；`g` = 起点到当前格的累计进入代价；
   `h` = 可采纳下界；`f = g + h`。堆内以 `(f, h, seq, node)` 排序，保证
   f 相同时优先扩展更接近目标的节点，且结果确定可复现。
2. **题目二**：`E` 进入代价 0，故朴素 `h = Manhattan(n,E)` 会高估最后一步，
   **不是可采纳启发函数**；本实现默认用 `h = max(0, w_min·(Manhattan−1))`
   （`w_min` 为地图最小地形代价），并保留 `manhattan` / `weighted` 供对照。
   `experiments.py` 会搜索并打印一个朴素曼哈顿启发给出次优解的随机反例。
3. **题目三**：`TreeNode` 兼容两种输入——无标注时按层交替（根 MAX），
   带 `is_maximizing_player` 时以该字段为准；子节点默认从左到右访问。
4. **题目四**：用 A\* 预计算 A、B、T1..T4 两两最小代价得到 6×6 矩阵；
   `Gain(p,Ti)=Vi−d(p,Ti)`，`U=Score_A−Score_B`；状态含双方位置、剩余宝藏、
   累计收益、回合数（A/B 由 `ply` 奇偶决定）。递归返回「相对当前 U 的增量」，
   因此置换表键 `(posA,posB,remaining,ply)` 可跨不同绝对收益复用；
   α-β 因返回值可能只是窗口界，只在 Minimax 上启用缓存。
5. **统计口径**：A\* 的「搜索节点数」= `SearchStats.expanded`（有效出队并扩展的
   状态数，含目标状态）；博弈树的开销 = `GameStats.leaves`（叶评估数），
   `cutoffs` 为触发 β≤α 的次数，`skipped` 为被剪掉的动作数。

## 对照实验索引（详见 results.json）

- Q1：`h=0` / 曼哈顿 / 加权 A\* 在给定迷宫与自设 20×20 随机障碍迷宫上的扩展节点数。
- Q2：四种启发函数的代价与扩展节点数、代价累计过程、不可采纳启发导致次优解的随机反例。
- Q3：给定树与自设 4 层×4 分支深树上，Minimax / Alpha-Beta / 四种访问顺序的叶评估数与剪枝数。
- Q4：A\* 与 Dijkstra 预计算距离矩阵的开销、H=2/4 决策差异、五种访问顺序的剪枝效果、
  置换表命中、高代价地形 ×2 与 T1 价值改为 20 的拓展场景。
