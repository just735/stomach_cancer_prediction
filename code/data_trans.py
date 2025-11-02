import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --------------------------
# 步骤1：读取基因表达数据（series_matrix.txt）
# --------------------------
# 读取文件，跳过以"!"开头的注释行，以第一列（ID_REF，基因探针ID）为索引
expr_data = pd.read_csv(
    "GSE15459_series_matrix.txt",
    sep="\t",
    comment="!",
    index_col="ID_REF"
)

# 数据转置：行→样本（GSM编号），列→基因（探针ID），适配"样本×特征"的建模格式
expr_data_T = expr_data.T

# 重置索引：将"样本GSM编号"从索引转为列（命名为"sample_id"），便于后续与临床数据匹配
expr_data_T = expr_data_T.reset_index().rename(columns={"index": "sample_id"})

# 筛选合格样本：排除质量控制失败或非胃癌腺癌的样本（参考文件中!Sample_data_processing标注）
excluded_samples = [
    "GSM387788", "GSM387790", "GSM387793", "GSM387797",
    "GSM387798", "GSM387799", "GSM387842"
]
expr_data_clean = expr_data_T[~expr_data_T["sample_id"].isin(excluded_samples)].copy()

# --------------------------
# 步骤2：读取临床信息（outcome.xls）并提取特征
# --------------------------
# 读取Excel格式的临床文件
clinical_data = pd.read_excel("GSE15459_outcome.xls")

# 提取核心临床列
clinical_core = clinical_data[["GSM ID", "Age_at_surgery", "Gender", "Laurenclassification",
                               "Stage", "Overall.Survival (Months)**", "Outcome (1=dead)"]].rename(
    columns={"GSM ID": "sample_id"})

# 处理性别和Lauren分类的分类变量，使用LabelEncoder进行编码
le = LabelEncoder()
clinical_core['Gender'] = le.fit_transform(clinical_core['Gender'])
clinical_core['Laurenclassification'] = le.fit_transform(clinical_core['Laurenclassification'])

# --------------------------
# 步骤3：合并表达数据与临床数据
# --------------------------
# 按样本ID（sample_id）内连接，仅保留同时有表达数据和临床数据的样本
merged_data = pd.merge(
    expr_data_clean,
    clinical_core,
    on="sample_id",
    how="inner"
)

# 去除sample_id列，因为它对模型训练没有帮助
merged_data = merged_data.drop("sample_id", axis=1)

# --------------------------
# 步骤4：准备特征和标签，划分训练集和测试集
# --------------------------
# 准备特征和标签
X = merged_data.drop("Outcome (1=dead)", axis=1)
y = merged_data["Outcome (1=dead)"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 步骤5：模型训练和评估
# --------------------------
# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练模型
rf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"模型准确率: {accuracy}")
print(f"模型F1分数: {f1}")
print(f"模型AUC值: {auc}")