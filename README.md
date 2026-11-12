# Contemporary_Artificial_Intelligence_Lab

本仓库保存《当代人工智能》课程的实验代码、运行说明及实验结果。

## 仓库结构

```text
Contemporary_Artificial_Intelligence_Lab/
├── README.md
├── Project_1/                 # 实验一：文本分类
│   └── README.md
└── Project_2/                 # 实验二：搜索算法
    └── README.md
```

各实验的依赖、输入格式、配置和完整运行步骤见对应目录的 README。下面的命令均从仓库根目录开始执行。

## Project_1：文本分类

- **任务**：新闻组文本 10 分类
- **方法**：TF-IDF + 逻辑回归 / 线性 SVM
- **输出**：`predictions.csv`、`loss_lr.png`、`loss_svm.png`
- **详细说明**：[Project_1/README.md](Project_1/README.md)

```bash
cd Project_1
pip install -r requirements.txt
python main_experiment.py
```

## Project_2：搜索算法

- **任务**：迷宫寻路、加权地形寻路、博弈树求值、竞争环境下的宝藏分配
- **方法**：A*、Minimax、Alpha-beta 剪枝，并比较启发函数、搜索深度、访问顺序及缓存的影响
- **输出**：四题求解结果；`results/raw/` 原始记录、`results/tables/` 结果表、`results/figures/` 图表
- **详细说明**：[Project_2/README.md](Project_2/README.md)

运行四题示例和正确性检查，仅需 Python 标准库：

```bash
cd Project_2
python main.py
python test_correctness.py
```

在 `Project_2` 目录中运行完整对照实验和绘图：

```bash
pip install -r requirements.txt
python -m experiments.run_experiments
python -m experiments.plot_results
```

实验中的随机输入使用固定种子，具体设置见各实验说明。重复运行时，确定性结果应保持一致；耗时会随运行环境与系统负载变化。
