import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_validate, KFold, GridSearchCV,train_test_split
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

# 读取预处理后的数据文件

# 原来读取 data.csv 的部分替换为读取刚生成的 data_merged.csv 并更稳健地提取标签列
data_path = r"D:\code\Prediction-model-of-distant-metastasis-of-breast-cancer-main\data\data_merged.csv"
data = pd.read_csv(data_path, index_col=0)

# 优先使用明确的标签列名，否则退回到最后一列
label_col = "Outcome (1=dead)"
if label_col in data.columns:
    X = data.drop(columns=[label_col])
    y = data[label_col]
else:
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

# 如果标签是字符串，尝试转换为数值（常见情况：1/0 或 dead/alive 等）
if y.dtype == object:
    try:
        y = y.astype(int)
    except Exception:
        y = y.map(lambda v: 1 if str(v).lower() in ("1", "dead", "metastasis", "yes", "positive", "true") else 0)

# 保持类别比例，使用 stratify（若类别>1）
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15,
    stratify=y if len(pd.Series(y).unique()) > 1 else None
)

n_features = X_train.shape[1]
max_feat_options = ["sqrt", "log2", None, 0.5]  # 常用合法选项
# 仅在特征数足够时加入整数选项
for v in (16, 21):
    if v <= n_features:
        max_feat_options.append(v)
# 去重并保持顺序
from collections import OrderedDict
max_feat_options = list(OrderedDict.fromkeys(max_feat_options))

# 简化的 max_features 选项
n_features = X_train.shape[1]
max_feat_options = ["sqrt", "log2", None, 0.5]
for v in (16, 21):
    if v <= n_features:
        max_feat_options.append(v)
from collections import OrderedDict
max_feat_options = list(OrderedDict.fromkeys(max_feat_options))

# 1) 特征选择（速度优先，调参时可增加 k）
k_select = min(1000, n_features)   # 将特征降到最多 1000，视内存/时间再调整
selector = SelectKBest(score_func=f_classif, k=k_select)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# 2) 随机搜索参数分布（比完全网格快很多）
param_dist = {
    "criterion": ["gini", "entropy"],
    "n_estimators": stats.randint(50, 151),    # 50-150 随机
    "max_depth": [3, 5, 7, None],
    "max_features": max_feat_options
}

reg = RFC(random_state=1412, verbose=0, n_jobs=1)  # 禁用内部并行
rnd = RandomizedSearchCV(
    estimator=reg,
    param_distributions=param_dist,
    n_iter=50,               # 随机试验次数，越大越慢
    scoring="roc_auc",
    cv=3,                    # 折数减少以加快
    verbose=0,
    n_jobs=8,
    random_state=1412
)

# 捕获中断或超时，给出提示
try:
    rnd.fit(X_train_sel, y_train)
except KeyboardInterrupt:
    print("训练被中断（KeyboardInterrupt）。建议：减小 n_iter、降低 k_select 或使用更少的 cv 折数后重试。")
    raise

# 把 search 指向 rnd，后续代码保持不变（使用 selector 变换的测试集）
search = rnd
# 预测（一次性获取），注意使用选择后的测试集
preds = search.predict(X_test_sel)

# 预测与概率
probs = None
if hasattr(search, "predict_proba"):
    probs = search.predict_proba(X_test_sel)
    if probs.shape[1] == 2:
        probs = probs[:, 1]
    else:
        probs = None

# 保存对全部样本的预测（可选）
X_all = X  # 全部特征，或使用 selector.transform(X) 如果做了特征选择
# 使用训练时的 selector 对全部样本做相同的特征选择，保证特征数一致
try:
    X_all_sel = selector.transform(X_all)
except Exception:
    # 如果 selector 不可用或转换失败，回退到原始 X_all（但会导致特征不匹配错误）
    X_all_sel = X_all.values if hasattr(X_all, "values") else X_all

preds_all = search.predict(X_all_sel)
probs_all = None
if hasattr(search, "predict_proba"):
    probs_ = search.predict_proba(X_all_sel)
    probs_all = probs_[:,1] if probs_.shape[1]==2 else None

# 输出目录（使用相对路径，确保存在）
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(out_dir, exist_ok=True)

# 构造输出表（保留测试集索引）
df_out = pd.DataFrame({"true": y_test, "pred": preds}, index=X_test.index)
if probs is not None:
    df_out["prob_positive"] = probs
csv_path = os.path.join(out_dir, "predictions.csv")
df_out.to_csv(csv_path, index=True)

# 保存全部样本预测
out_all = pd.DataFrame({"true": y, "pred": preds_all}, index=X_all.index)
if probs_all is not None:
    out_all["prob_positive"] = probs_all
out_all.to_csv(os.path.join(out_dir, "predictions_all.csv"), index=True)
print("已保存全部样本预测到: output/predictions_all.csv")

# 评估指标
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
cm = confusion_matrix(y_test, preds)

auc = None
roc_path = None
if probs is not None:
    # 使用 best_estimator_ 的类别顺序将真实标签二值化
    from pathlib import Path

    # 优先读取 R 脚本导出的 normalized_expression.csv + clinical_matched.csv，
    # 若不存在则回退读取 data/data_merged.csv
    BASE_DIR = Path(__file__).resolve().parents[1]
    expr_path = BASE_DIR / "data" / "normalized_expression.csv"
    clin_path = BASE_DIR / "data" / "clinical_matched.csv"
    merged_path = BASE_DIR / "data" / "data_merged.csv"

    if expr_path.exists() and clin_path.exists():
        expr = pd.read_csv(expr_path, index_col=0)
        clin = pd.read_csv(clin_path)
        if "sample_id" in clin.columns:
            data = expr.merge(clin, left_index=True, right_on="sample_id", how="inner").set_index("sample_id")
        else:
            clin_indexed = clin.set_index(clin.columns[0])
            data = expr.merge(clin_indexed, left_index=True, right_index=True, how="inner")
        print(f"Loaded expr+clin: {expr_path.name} + {clin_path.name}  -> samples={data.shape[0]}, features={data.shape[1]}")
    elif merged_path.exists():
        data = pd.read_csv(merged_path, index_col=0)
        print(f"Loaded merged data: {merged_path.name}  -> samples={data.shape[0]}, columns={data.shape[1]}")
    else:
        raise FileNotFoundError(
            "找不到数据文件：请先运行 code/generate_merged_data.py 生成 data/data_merged.csv，\n"
            "或将 R 脚本生成的 normalized_expression.csv 与 clinical_matched.csv 放到 data/ 目录下。"
        )
    classes = getattr(search.best_estimator_, "classes_", None)
    try:
        y_bin = label_binarize(y_test, classes=classes)[:, 1]
        auc = roc_auc_score(y_bin, probs)
        fpr, tpr, _ = roc_curve(y_bin, probs)
        # 保存 ROC 曲线
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
    except Exception:
        auc = None

# 简洁输出到控制台（仅关键信息）
print(f"预测已保存: {csv_path}")
print(f"准确率: {acc:.4f}")
if auc is not None:
    print(f"AUC: {auc:.4f}    ROC图: {roc_path}")
print("混淆矩阵:")
print(cm)

print("\n分类报告:")
print(report)

print("原始样本数:", data.shape[0])    # 如果有 data 变量
print("X 总行数:", X.shape[0])
print("X_train 行数:", X_train.shape[0])
print("X_test  行数:", X_test.shape[0])