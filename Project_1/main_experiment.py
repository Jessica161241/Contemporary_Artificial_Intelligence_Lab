# main_experiment.py
# 文本分类实验：逻辑回归 + 线性 SVM
# 逐 batch 记录训练损失，逐 epoch 记录验证损失，损失曲线波动下降

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss

from preprocess import clean_text

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_EPOCHS = 20
BATCH_SIZE = 128
ALPHA = 1e-4
ETA0 = 0.1


def load_data():
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    return (train_df['text'].astype(str).tolist(),
            train_df['target'].values,
            test_df['text'].astype(str).tolist())


def compute_hinge_loss(clf, X, y):
    """OvA hinge loss：对每个样本、每个类别 max(0, 1 - y_signed * score)。"""
    scores = clf.decision_function(X)
    n, k = scores.shape
    y_signed = np.full((n, k), -1.0)
    y_signed[np.arange(n), y] = 1.0
    return float(np.maximum(0.0, 1.0 - y_signed * scores).mean())


def train_sgd(X_tr, y_tr, X_val, y_val, loss_type):
    """
    通用 mini-batch SGD 训练循环。
    loss_type='log'   -> 逻辑回归，损失用 log_loss
    loss_type='hinge' -> 线性 SVM，损失用 hinge loss
    """
    clf = SGDClassifier(
        loss='log_loss' if loss_type == 'log' else 'hinge',
        penalty='l2',
        alpha=ALPHA,
        learning_rate='constant',
        eta0=ETA0,
        random_state=RANDOM_STATE
    )
    classes = np.unique(y_tr)
    train_losses, val_losses, val_marks = [], [], []
    n = X_tr.shape[0]
    rng = np.random.RandomState(RANDOM_STATE)
    step = 0

    def loss_on(X, y):
        if loss_type == 'log':
            return log_loss(y, clf.predict_proba(X), labels=classes)
        return compute_hinge_loss(clf, X, y)

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            X_b, y_b = X_tr[idx], y_tr[idx]
            if step == 0:
                clf.partial_fit(X_b, y_b, classes=classes)
            else:
                clf.partial_fit(X_b, y_b)
            train_losses.append(loss_on(X_b, y_b))
            step += 1

        val_losses.append(loss_on(X_val, y_val))
        val_marks.append(step)
        tag = 'LR' if loss_type == 'log' else 'SVM'
        print(f"[{tag}] Epoch {epoch+1:02d}/{N_EPOCHS}  "
              f"TrainLoss(batch)={train_losses[-1]:.4f}  "
              f"ValLoss={val_losses[-1]:.4f}")

    return clf, train_losses, val_losses, val_marks


def plot_curve(train_losses, val_losses, val_marks, title, save_path):
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(train_losses)), train_losses,
             label='Train Loss (per batch)', alpha=0.55, linewidth=0.9)
    plt.plot(val_marks, val_losses, label='Val Loss (per epoch)',
             linewidth=2.0, marker='o', markersize=4)
    plt.xlabel('Batch Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"已保存 {save_path}")


def main():
    print("===== 1. 加载数据 =====")
    X, y, X_test = load_data()
    print(f"训练集: {len(X)}, 测试集: {len(X_test)}")

    print("\n===== 2. 文本清洗 =====")
    X_clean = [clean_text(t) for t in X]
    X_test_clean = [clean_text(t) for t in X_test]
    print("示例：", X_clean[0][:150])

    print("\n===== 3. 划分训练 / 验证 (8:2, stratify) =====")
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

    print("\n===== 4. TF-IDF 特征提取 =====")
    vec = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        stop_words='english', sublinear_tf=True, min_df=2
    )
    X_train_tfidf = vec.fit_transform(X_train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test_clean)
    print(f"特征维度: {X_train_tfidf.shape[1]}")

    print("\n===== 5. 逻辑回归 (SGD, log loss) =====")
    lr_clf, lr_train, lr_val, lr_marks = train_sgd(
        X_train_tfidf, y_train, X_val_tfidf, y_val, loss_type='log'
    )
    lr_train_acc = accuracy_score(y_train, lr_clf.predict(X_train_tfidf))
    lr_pred = lr_clf.predict(X_val_tfidf)
    acc_lr = accuracy_score(y_val, lr_pred)
    f1_lr = f1_score(y_val, lr_pred, average='macro')
    print(f"\nLR 训练准确率={lr_train_acc:.4f}  验证准确率={acc_lr:.4f}  "
          f"差距={lr_train_acc - acc_lr:.4f}")
    print(f"LR 宏F1={f1_lr:.4f}")
    print(classification_report(y_val, lr_pred))

    print("\n===== 6. 线性 SVM (SGD, hinge loss) =====")
    svm_clf, svm_train, svm_val, svm_marks = train_sgd(
        X_train_tfidf, y_train, X_val_tfidf, y_val, loss_type='hinge'
    )
    svm_train_acc = accuracy_score(y_train, svm_clf.predict(X_train_tfidf))
    svm_pred = svm_clf.predict(X_val_tfidf)
    acc_svm = accuracy_score(y_val, svm_pred)
    f1_svm = f1_score(y_val, svm_pred, average='macro')
    print(f"\nSVM 训练准确率={svm_train_acc:.4f}  验证准确率={acc_svm:.4f}  "
          f"差距={svm_train_acc - acc_svm:.4f}")
    print(f"SVM 宏F1={f1_svm:.4f}")
    print(classification_report(y_val, svm_pred))

    print("\n===== 7. 绘制损失曲线 =====")
    plot_curve(lr_train, lr_val, lr_marks,
               'Logistic Regression (SGD, log loss)', 'loss_lr.png')
    plot_curve(svm_train, svm_val, svm_marks,
               'Linear SVM (SGD, hinge loss)', 'loss_svm.png')

    print("\n===== 8. 最优模型在全量数据上重训并预测 =====")
    best = 'LR' if acc_lr >= acc_svm else 'SVM'
    print(f"选择: {best} (LR={acc_lr:.4f}, SVM={acc_svm:.4f})")

    vec_full = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        stop_words='english', sublinear_tf=True, min_df=2
    )
    X_full = vec_full.fit_transform(X_clean)
    X_test_full = vec_full.transform(X_test_clean)

    final = SGDClassifier(
        loss='log_loss' if best == 'LR' else 'hinge',
        penalty='l2', alpha=ALPHA,
        learning_rate='constant', eta0=ETA0,
        max_iter=N_EPOCHS, random_state=RANDOM_STATE
    )
    final.fit(X_full, y)
    y_pred = final.predict(X_test_full)

    pd.DataFrame(y_pred).to_csv('predictions.csv', index=False, header=False)
    print("-> predictions.csv")
    print("标签分布：")
    print(pd.Series(y_pred).value_counts().sort_index())


if __name__ == "__main__":
    main()