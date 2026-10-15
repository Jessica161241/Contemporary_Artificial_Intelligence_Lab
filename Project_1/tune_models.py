# tune_models.py
# 1) 对齐 TUNING.md：标准库模型调 C
# 2) 为损失可视化：SGD 版模型调 alpha / eta0

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from preprocess import clean_text

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def build_data():
    df = pd.read_csv("train_data.csv")
    X = [clean_text(t) for t in df['text'].astype(str).tolist()]
    y = df['target'].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    vec = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        stop_words='english', sublinear_tf=True, min_df=2
    )
    return vec.fit_transform(X_tr), y_tr, vec.transform(X_val), y_val


def tune_standard_c(X_tr, y_tr, X_val, y_val):
    """
    对齐官方 TUNING.md 的调参建议，用标准库模型调 C。
    说明：LogisticRegression / LinearSVC 的 C 与 SGDClassifier 的 alpha
    互为倒数关系：alpha = 1 / (C * n_samples)，正则化强度等价。
    """
    print("===== 标准库模型 C 参数对比 (对齐 TUNING.md) =====")
    print(f"{'C':<8}{'LR Acc':<10}{'LR F1':<10}"
          f"{'SVM Acc':<10}{'SVM F1':<10}")
    for C in [0.1, 1, 10, 100]:
        lr = LogisticRegression(C=C, max_iter=2000,
                                random_state=RANDOM_STATE)
        lr.fit(X_tr, y_tr)
        lr_pred = lr.predict(X_val)

        svm = LinearSVC(C=C, max_iter=3000, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_tr)
        svm_pred = svm.predict(X_val)

        print(f"{C:<8}"
              f"{accuracy_score(y_val, lr_pred):<10.4f}"
              f"{f1_score(y_val, lr_pred, average='macro'):<10.4f}"
              f"{accuracy_score(y_val, svm_pred):<10.4f}"
              f"{f1_score(y_val, svm_pred, average='macro'):<10.4f}")
    print()


def tune_sgd(X_tr, y_tr, X_val, y_val, loss_type):
    """
    SGD 版模型调 alpha / eta0。用于损失可视化时，选出最合适的
    SGD 超参数。alpha 与标准库 C 等价（倒数关系）。
    """
    name = 'SGD 逻辑回归 (log loss)' if loss_type == 'log_loss' \
        else 'SGD 线性 SVM (hinge)'
    print(f"===== {name} 参数对比 =====")
    print(f"{'alpha':<10}{'eta0':<8}{'Acc':<10}{'F1':<10}")
    for alpha in [1e-4, 1e-3, 1e-2]:
        for eta0 in [0.001, 0.01, 0.05, 0.1]:
            clf = SGDClassifier(
                loss=loss_type, penalty='l2', alpha=alpha,
                learning_rate='constant', eta0=eta0,
                max_iter=30, random_state=RANDOM_STATE
            )
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_val)
            print(f"{alpha:<10}{eta0:<8}"
                  f"{accuracy_score(y_val, pred):<10.4f}"
                  f"{f1_score(y_val, pred, average='macro'):<10.4f}")
    print()


if __name__ == "__main__":
    X_tr, y_tr, X_val, y_val = build_data()
    tune_standard_c(X_tr, y_tr, X_val, y_val)
    tune_sgd(X_tr, y_tr, X_val, y_val, 'log_loss')
    tune_sgd(X_tr, y_tr, X_val, y_val, 'hinge')