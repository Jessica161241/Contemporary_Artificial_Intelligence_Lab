# 实验二：A*、Minimax 与 Alpha-beta 搜索

本项目独立实现四道题的算法，并保存可复现的正确性检查、对照实验、原始数据和图表。

## 1. 环境与依赖

实测环境：Windows 11、Python 3.13.9、Matplotlib 3.9.2、NumPy 2.5.4。完整信息见 `results/raw/environment.json`。算法和正确性检查仅依赖 Python 标准库；绘图和完整实验使用后两项依赖。代码按跨平台路径编写，但目前只在 Windows 上实际运行验证。

在终端进入本目录后安装绘图依赖：

```bash
pip install -r requirements.txt
```

需要使用相同绘图环境时，可安装实测版本：

```bash
pip install matplotlib==3.9.2 numpy==2.5.4
```

推荐 Python 3.10 或更新版本；其他版本与平台未逐一测试。无须网络数据集、模型下载、GPU 或机器学习框架。

## 2. 数据与目录

```text
Project_2/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                      # 四题运行入口
├── test_correctness.py          # 正确性检查
├── algorithms/
│   ├── astar.py                 # A* 搜索
│   ├── minimax.py               # Minimax 搜索
│   └── alphabeta.py             # Alpha-beta 剪枝
├── problems/
│   ├── maze.py                  # 迷宫规则与示例
│   ├── terrain.py               # 加权地形规则与示例
│   ├── game_tree.py             # 博弈树结构与示例
│   └── treasure.py              # 宝藏地图、状态与收益
├── examples/
│   ├── maze.json                # 题一输入
│   ├── terrain.json             # 题二输入
│   └── tree.json                # 题三输入
├── experiments/
│   ├── scenarios.py             # 自设场景
│   ├── run_experiments.py       # 运行对照实验
│   └── plot_results.py          # 生成图表
└── results/
    ├── raw/                    # 原始数据与运行环境
    ├── tables/                 # 七份 CSV 结果表
    └── figures/                # 五张结果图
```

不需要另外下载数据。直接运行 `main.py` 时，程序读取 `problems/` 中的内置示例；指定 `--input` 时，读取对应的 JSON 文件。输入文件放在 `examples/` 中，也可以通过路径指定其他位置。

| 题目 | 输入文件或位置 | 字段与内容 |
|---|---|---|
| 一：迷宫 | `examples/maze.json`；内置示例为 `problems/maze.py` 中的 `MAZE` | `start`、`goal` 为坐标数组；`grid` 为由 S、E、`.`、X 组成的二维数组 |
| 二：加权地形 | `examples/terrain.json`；内置示例为 `problems/terrain.py` 中的 `TERRAIN` | `start`、`goal` 为坐标数组；`grid` 保存地形权值和 S、E 标记，可用 X 表示障碍 |
| 三：博弈树 | `examples/tree.json`；内置示例由 `problems/game_tree.py` 中的 `sample_tree()` 构造 | `tree_structure` 为根；非叶节点用 `children` 保存子节点，叶节点用 `value` 保存得分；`is_maximizing_player` 表示 MAX/MIN |
| 四：宝藏分配 | `problems/treasure.py`，无需单独输入文件 | `TREASURE_MAP` 为地图，`POSITIONS` 为双方与宝藏位置，`VALUES` 为宝藏价值；深度在调用时指定 |

JSON 中的 S、E、X 等字符需要加引号。题三左右两侧各有一组 `[6,9]`，按树中不同路径分别访问，不作节点去重。随机测试和拓展实验的输入由脚本按固定种子生成，不需要提前准备文件。

坐标统一为零起点 `(row, col)`。二维输入必须非空、矩形，坐标须在范围内。

| 题目 | 函数入口 | 输入与返回值 |
|---|---|---|
| 一 | `problems.maze.find_path(start, goal, grid)` | `S/E/.` 可通行，`X` 阻挡；返回含起终点的元组坐标列表，无解 `[]` |
| 二 | `problems.terrain.minimum_cost(start, goal, grid)` | 数字或整数字符串表示非负代价；S=1、E=0、X 阻挡；返回整数最小代价，无解 `-1` |
| 三 | `problems.game_tree.best_score(root)` | `TreeNode(value=None, children=None)`，根 MAX；返回最优值 |
| 四 | `problems.treasure.distance_matrix()`、`solve_treasure(...)` | 给定地图、宝藏价值；返回距离矩阵及带指标的搜索结果 |

题二起点代价不计，进入 E 免费；题四 A/B/T 格均为 1，采用独立配置。算法内部的无解代价为 `inf`，对外接口转换为规定格式。

## 3. 运行顺序与预期输出

以下命令均在 `Project_2` 目录执行。

### 3.1 正确性检查

```bash
python test_correctness.py
```

预期 8 项测试全部 `OK`。除了四题示例，还检查无解、单格、免费终点等边界情况，并自动生成额外输入：

- 300 张随机地图：与另一份 Dijkstra 实现比较最小代价，并对其中 200 张检查路径是否合法。
- 100 棵随机树：与直接递归计算的结果比较。
- 20 组五宝藏案例：在不同深度、顺序和缓存设置下搜索，共检查 240 次，与穷举结果比较。

这些测试用于检查算法能否处理示例之外的情况，例如路径代价算错、剪枝出错或缓存改变最优结果。测试中的参照实现不调用被检查的搜索函数，避免相同错误被重复使用。这部分用于检查正确性，性能对照另由 `experiments/` 完成。

### 3.2 四题测试样例

```bash
python main.py
python main.py --problem 1
python main.py --problem 2
python main.py --problem 3
python main.py --problem 4
```

题一默认输出：

```text
[(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
```

题二输出 `7`；题三输出 `5`。题四先输出带 A/B/T1/T2/T3/T4 标签的 6×6 距离矩阵，再分别输出 Minimax、Alpha-beta 在 H=2 与 H=4 下的结果；两种算法均为：

```json
{"best_target": "T2", "best_value": -1}
{"best_target": "T3", "best_value": 3}
```

实际命令输出另带 `method`、`H` 字段以区分算法和深度。

### 3.3 自定义输入

```bash
python main.py --problem 1 --input examples/maze.json
python main.py --problem 2 --input examples/terrain.json
python main.py --problem 3 --input examples/tree.json
```

题一、题二 JSON 字段为 `start`、`goal`、`grid`，坐标用 JSON 数组。题三使用 `tree_structure` 格式；玩家标志如提供，必须从 MAX 开始逐层交替。题四命令行使用内置示例输入；扩展场景通过实验模块和 Python API 运行。

### 3.4 生成对照数据和图表

```bash
python -m experiments.run_experiments
python -m experiments.plot_results
```

第一步覆盖 `results/raw/`、`results/tables/` 对应产物；第二步覆盖五张图，不修改报告。普通电脑上通常数秒到数十秒，依机器负载变化。实验脚本内置一致性断言，若不同算法的最优值不一致会中止。

## 4. 配置与公平比较

- A* 使用 `heapq`、严格改善才松弛、惰性删除过期堆项，并允许更低 g 重新展开。邻居固定为右、下、左、上，同 f 使用 FIFO。
- 题一比较 `h=0` 和曼哈顿距离；题二比较 `h=0` 和 `c_min * max(Manhattan-1, 0)`。普通地形含零代价时可退化为零启发。
- `raw_manhattan` 仅作为启发函数高估的研究开关，不用于正式答案或性能主表；题二不能无条件将其当作可采纳启发。
- Minimax 与 Alpha-beta 共用同一状态、行动和评价规则。题三正式接口从左到右，逆序仅用于对照。
- 题四默认自然顺序 T1→T2→T3→T4；逆序为反向；gain 排序按当前行动者的即时净收益降序，相等时按编号。只在严格改善时更换首步，因此不同顺序遇到并列最优时可返回不同的合法最优首步。
- H 是行动次数而非双方各走一轮。收益 `Vi-distance` 可负，不允许主动跳过。H 截断时评价当前累计收益差，不添加未经要求的未来估计。
- 缓存键包括双方位置、剩余掩码、双方累计收益、当前玩家、剩余深度及 MAX/MIN 标志；只缓存整个后代均完整搜索的精确值。缓存按单次搜索新建，预热不复用缓存。
- 网格尺寸为 10/30/60，种子 42–46；障碍率分别为 0.27（墙）与 0.20（加权），加权值从 `(1,1,1,3,8)` 抽样，顶行和最右列设为可行路线。
- 扩展博弈为 4/6/8 个宝藏、H=n，固定种子 42，在 20×20 无障碍单位网格抽取互异坐标。其距离直接用已知精确的曼哈顿距离，目的是隔离博弈搜索规模，不将它计为额外 A* 基准。
- 每个配置预热 1 次，重复 7 次取墙钟时间中位数；保存全部时间样本。单次计时包围求解函数，包括函数内建模、排序与缓存，不含绘图、文件 I/O；题四距离预计算独立记录。小样例计时仅作参考。

## 5. 指标口径

| 字段 | 定义 |
|---|---|
| `expanded` | A* 有效出堆次数，包含起点和终点，不含过期堆项；重新展开另计 |
| `generated` | 初始堆项加成功松弛后新入堆的次数，不是邻居检查次数 |
| `frontier_peak` | 堆中条目峰值，包含尚未弹出的过期项，不等于唯一节点数 |
| `visited` | 博弈递归调用次数，包括根、叶子和缓存命中的调用 |
| `evaluated` | 实际执行叶子/截断状态评价的次数；缓存命中不重复评价 |
| `cutoffs` | 确有剩余兄弟未访问时触发的剪枝事件数 |
| `skipped_children` | 被直接跳过的兄弟边数，不是整棵被省略子树的节点数 |
| `cache_hits/entries` | 精确值缓存的命中次数/条目数；均为次数或规模代理，不是内存字节 |
| `time_median_ms` | 7 次端到端求解计时中位数，单位毫秒 |

## 6. 结果文件与核验

`results/raw/experiments.json` 包含所有计时样本、示例路径与展开序列、题三剪枝轨迹、根行动值、代表性博弈主线、扩展输入和指标。七张 CSV 分别为 `distance_matrix`、`sample_search`、`tree_search`、`treasure_search`、`root_action_values`、`grid_scaling`、`game_scaling`。

主要结果：题一展开数 16→14，题二 15→11；题四 H=4 的访问数为 Minimax 65、普通 Alpha-beta 47、收益排序 37。确定性节点数应复现，耗时无需逐位相同。重新运行实验后，报告中的静态计时需同步更新；图表会从最新原始数据重建。

## 7. 限制与失败情况

本实验支持矩形有限网格、非负整数进入代价、有限且角色交替的博弈树。负权、非矩形网格、越界坐标等非法输入抛出 `ValueError`；合法但不可达地图使用题目规定的无解值。题四示例和扩展场景中的各个位置均互相可达；扩展接口过滤不可达目标，但若仍有宝藏而当前玩家无合法行动，会抛出异常，不擅自定义跳过或提前终局。

缓存是保守精确值方案，未实现上下界置换表及收益差归一化，因此命中率有限。扩展博弈每个规模仅一组地图，不能据此宣称排序普遍最优。重复计时非独立机器实验，未做 CPU 绑核和显著性检验。报告以节点数量和机制分析为主要证据。

## 8. 运行检查

运行测试和四题示例后，核对结果是否与上文一致。绘图完成后，检查 `results/figures/` 中的五张图片能否正常打开。重新运行实验会更新原始数据和图表，报告中的计时表需相应更新。
