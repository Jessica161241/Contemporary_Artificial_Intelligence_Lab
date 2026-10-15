# 文本分类实验 — 新闻组 10 分类

## 环境配置
```bash
pip install -r requirements.txt
```

## 文件结构
| 文件 | 说明 |
|---|---|
| `preprocess.py` | 文本清洗（去邮件头、提取 Subject、去引用行/URL/邮箱） |
| `main_experiment.py` | 主实验：训练逻辑回归与线性 SVM，画损失曲线，生成预测 |
| `tune_models.py` | 超参数对比：标准库 C 参数 + SGD 版 alpha/eta0 |
| `data_exploration.py` | 数据初步探索 |
| `train_data.csv` | 训练数据 |
| `test_data_unlabeled.csv` | 测试数据 |
| `predictions.csv` | 输出：测试集预测标签 |
| `loss_lr.png` | 输出：逻辑回归损失曲线 |
| `loss_svm.png` | 输出：线性 SVM 损失曲线 |

## 运行步骤
1. 数据探索（可选，用于了解数据内容和结构）
```bash
python data_exploration.py
```
2. 超参数对比（可选，用于调参）
```bash
python tune_models.py
```
3. 主实验（生成损失曲线与 `predictions.csv`）
```bash
python main_experiment.py
```

## 复现说明
- 全局随机种子 `RANDOM_STATE = 42`，所有随机操作均已固定。
- 训练/验证划分 8:2 分层抽样。
- 本地两次运行结果一致。

## 主要超参数
| 参数 | 值 |
|---|---|
| TF-IDF max_features | 10000 |
| ngram_range | (1, 2) |
| sublinear_tf | True |
| min_df | 2 |
| 训练轮数 N_EPOCHS | 20 |
| batch_size | 128 |
| alpha | 1e-4 |
| eta0 | 0.1 |

## 求解器说明
- `main_experiment.py` 使用 `SGDClassifier`，以支持逐 batch 损失可视化。
- `tune_models.py` 同时使用标准库 `LogisticRegression` / `LinearSVC`（按 TUNING.md 调 `C`）与 `SGDClassifier`（调 `alpha/eta0`），两套实现互相对照。
- `SGDClassifier` 的 `alpha` 与标准库 `C` 满足 `alpha = 1 / (C · n_samples)`。
```
