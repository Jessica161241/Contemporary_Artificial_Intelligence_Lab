
import pandas as pd

# 1. 读取原始数据
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data_unlabeled.csv")

# 2. 查看数据的基本规模
print("===== 数据基本信息 =====")
print("训练集大小：", train_df.shape)
print("测试集大小：", test_df.shape)
print("训练集列名：", train_df.columns.tolist())
print("测试集列名：", test_df.columns.tolist())

# 3. 检查缺失值
print("\n===== 缺失值检查 =====")
print("训练集各列缺失值：")
print(train_df.isnull().sum())

print("\n测试集各列缺失值：")
print(test_df.isnull().sum())

# 4. 查看新闻类别的分布
print("\n===== 新闻类别分布 =====")
print(train_df["target"].value_counts().sort_index())

# 5. 查看新闻文本长度
train_df["text_length"] = train_df["text"].astype(str).str.len()

print("\n===== 新闻文本长度统计 =====")
print(train_df["text_length"].describe())

# 6. 查看几条原始新闻
print("\n===== 新闻文本示例 =====")
for i, text in enumerate(train_df["text"].head(3), start=1):
    print(f"\n第 {i} 条新闻：")
    print(str(text)[:300])


# 7. 检查新闻文本中邮件头部与正文的分隔情况
print("\n===== 邮件头部格式检查 =====")

for i, text in enumerate(train_df["text"].head(10), start=1):
    text = str(text)
    print(f"\n第 {i} 条新闻：")

    if "\n\n" in text:
        header, body = text.split("\n\n", 1)
        print("发现空行分隔符：是")
        print("头部前 200 个字符：")
        print(header[:200])
        print("正文前 200 个字符：")
        print(body[:200])
    else:
        print("发现空行分隔符：否")
        print("文本前 300 个字符：")
        print(text[:300])
