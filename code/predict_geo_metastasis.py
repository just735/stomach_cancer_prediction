"""
GEO胃癌转移预测模型
参考 predict_stad_metastasis.py 和 Prediction model of distant metastasis of stomach cancer.py 的写法
"""

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
from sklearn.model_selection import cross_validate, KFold, GridSearchCV, train_test_split
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from pathlib import Path

# 设置matplotlib后端
mlp.use('Agg')  # 使用非交互式后端，避免显示问题

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass  # 如果字体设置失败，继续执行

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "GEO"
OUTPUT_DIR = BASE_DIR / "output" / "GEO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GEO胃癌转移预测模型")
print("=" * 60)

# 读取预处理后的数据文件
data_path = DATA_DIR / "geo_processed.csv"
if not data_path.exists():
    print(f"\n错误: 数据文件不存在: {data_path}")
    print("请先运行 code/process_geo_data.py 生成处理后的数据")
    raise FileNotFoundError(f"数据文件不存在: {data_path}")

print(f"\n读取数据: {data_path}")
data = pd.read_csv(data_path, index_col=0)

# 提取特征和标签
label_col = "label"
if label_col not in data.columns:
    # 如果没有label列，尝试使用最后一列或Outcome列
    if "Outcome (1=dead)" in data.columns:
        label_col = "Outcome (1=dead)"
        print("使用 'Outcome (1=dead)' 作为标签")
    else:
        print("警告: 未找到'label'列，使用最后一列作为标签")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        label_col = None
else:
    label_col = "label"

if label_col:
    # 排除标签列和ID列（这些不能作为特征）
    exclude_cols = [label_col, 'case_id', 'submitter_id']
    # 也排除文本类型的分类特征（如果存在）
    text_cols = ['primary_diagnosis', 'morphology', 'tumor_stage']
    exclude_cols.extend([col for col in text_cols if col in data.columns])
    
    X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
    y = data[label_col]
    
    print(f"排除的列: {[col for col in exclude_cols if col in data.columns]}")

print(f"数据形状: {data.shape[0]} 个样本, {X.shape[1]} 个特征")
print(f"特征列数: {X.shape[1]} (前10个: {list(X.columns[:10])})")

# 处理分类变量：将字符串类型的特征编码为数值
from sklearn.preprocessing import LabelEncoder

print("\n处理分类变量...")
categorical_cols = []
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        categorical_cols.append(col)
        print(f"  编码列: {col}")
        le = LabelEncoder()
        # 处理缺失值：先用特殊值填充，编码后再恢复为NaN
        mask = X[col].isna()
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col].astype(str))
        X.loc[mask, col] = np.nan
        print(f"    唯一值: {list(le.classes_)}")

if len(categorical_cols) == 0:
    print("  无分类变量需要编码")

# 确保所有特征都是数值型
X = X.select_dtypes(include=[np.number])

# 处理缺失值：用中位数填充
print("\n处理缺失值...")
missing_counts = X.isna().sum()
if missing_counts.sum() > 0:
    print("  缺失值统计:")
    for col in missing_counts[missing_counts > 0].index[:10]:  # 只显示前10个
        print(f"    {col}: {missing_counts[col]} ({missing_counts[col]/len(X)*100:.1f}%)")
    if missing_counts.sum() > 10:
        print(f"    ... 还有 {len(missing_counts[missing_counts > 0]) - 10} 个列有缺失值")
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    print("  已用中位数填充缺失值")
else:
    print("  无缺失值")

print(f"\n最终特征列数: {X.shape[1]}")
print(f"最终数据形状: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

# 如果标签是字符串，尝试转换为数值
if y.dtype == object:
    try:
        y = y.astype(int)
    except Exception:
        y = y.map(lambda v: 1 if str(v).lower() in ("1", "metastasis", "yes", "positive", "true", "m1", "dead") else 0)

# 检查标签分布
label_counts = y.value_counts()
print(f"\n标签分布:")
print(f"  无转移 (0): {label_counts.get(0, 0)}")
print(f"  有转移 (1): {label_counts.get(1, 0)}")

# 如果类别不平衡，给出警告
if len(label_counts) == 2:
    ratio = min(label_counts) / max(label_counts)
    if ratio < 0.3:
        print(f"  警告: 类别不平衡严重 (比例 {ratio:.2f})，建议使用类别权重")

# 保持类别比例，使用 stratify（若类别>1）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15,
    stratify=y if len(pd.Series(y).unique()) > 1 else None
)

print(f"\n训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

n_features = X_train.shape[1]
max_feat_options = ["sqrt", "log2", None, 0.5]  # 常用合法选项
# 仅在特征数足够时加入整数选项
for v in (16, 21, 50, 100, 500, 1000):
    if v <= n_features:
        max_feat_options.append(v)
# 去重并保持顺序
from collections import OrderedDict
max_feat_options = list(OrderedDict.fromkeys(max_feat_options))

# 特征选择（如果特征数较多）
selector = None
X_train_sel = X_train.values
X_test_sel = X_test.values

if n_features > 10:
    k_select = min(1000, n_features)  # 将特征降到最多 1000
    print(f"\n特征选择: 从 {n_features} 个特征中选择前 {k_select} 个...")
    selector = SelectKBest(score_func=f_classif, k=k_select)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(f"  特征选择完成: {X_train_sel.shape[1]} 个特征")
else:
    print(f"\n特征数较少 ({n_features})，跳过特征选择")

# 随机搜索参数分布
param_dist = {
    "criterion": ["gini", "entropy"],
    "n_estimators": stats.randint(50, 151),  # 50-150 随机
    "max_depth": [3, 5, 7, None],
    "max_features": max_feat_options,
    "class_weight": ["balanced", None]  # 处理类别不平衡
}

reg = RFC(random_state=1412, verbose=0, n_jobs=1)  # 禁用内部并行
rnd = RandomizedSearchCV(
    estimator=reg,
    param_distributions=param_dist,
    n_iter=50,  # 随机试验次数
    scoring="roc_auc",
    cv=3,  # 折数
    verbose=1,
    n_jobs=8,
    random_state=1412
)

# 捕获中断或超时，给出提示
print("\n开始训练模型...")
try:
    rnd.fit(X_train_sel, y_train)
    print("模型训练完成！")
except KeyboardInterrupt:
    print("训练被中断（KeyboardInterrupt）。建议：减小 n_iter、降低 k_select 或使用更少的 cv 折数后重试。")
    raise

# 把 search 指向 rnd
search = rnd

# 预测（一次性获取）
preds = search.predict(X_test_sel)

# 预测与概率
probs = None
if hasattr(search, "predict_proba"):
    probs = search.predict_proba(X_test_sel)
    if probs.shape[1] == 2:
        probs = probs[:, 1]
    else:
        probs = None

# 保存对全部样本的预测
print("\n保存预测结果...")
try:
    if selector is not None:
        X_all_sel = selector.transform(X)
    else:
        X_all_sel = X.values
except Exception:
    X_all_sel = X.values

preds_all = search.predict(X_all_sel)
probs_all = None
if hasattr(search, "predict_proba"):
    probs_ = search.predict_proba(X_all_sel)
    probs_all = probs_[:, 1] if probs_.shape[1] == 2 else None

# 保存预测结果
df_out = pd.DataFrame({"true": y_test, "pred": preds}, index=X_test.index)
if probs is not None:
    df_out["prob_positive"] = probs
csv_path = OUTPUT_DIR / "predictions.csv"
df_out.to_csv(csv_path, index=True)
print(f"  测试集预测已保存: {csv_path}")

# 保存全部样本预测
out_all = pd.DataFrame({"true": y, "pred": preds_all}, index=X.index)
if probs_all is not None:
    out_all["prob_positive"] = probs_all
out_all_path = OUTPUT_DIR / "predictions_all.csv"
out_all.to_csv(out_all_path, index=True)
print(f"  全部样本预测已保存: {out_all_path}")

# 评估指标
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
cm = confusion_matrix(y_test, preds)

auc = None
roc_path = None
if probs is not None:
    classes = getattr(search.best_estimator_, "classes_", None)
    try:
        # 检查测试集中是否有两个类别
        unique_labels = pd.Series(y_test).unique()
        if len(unique_labels) < 2:
            print(f"  警告: 测试集中只有一个类别 ({unique_labels})，无法生成ROC曲线")
            auc = None
        elif len(classes) == 2:
            # 确保两个类别都存在
            y_bin = label_binarize(y_test, classes=classes)
            # 检查label_binarize的结果维度
            if y_bin.shape[1] == 2:
                y_bin = y_bin[:, 1]
            elif y_bin.shape[1] == 1:
                y_bin = y_test.values
            else:
                y_bin = y_test.values
            
            # 确保y_bin和probs的长度一致
            if len(y_bin) == len(probs):
                auc = roc_auc_score(y_bin, probs)
                fpr, tpr, _ = roc_curve(y_bin, probs)
                
                # 保存 ROC 曲线
                roc_path = OUTPUT_DIR / "roc_curve.png"
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
                plt.plot([0, 1], [0, 1], "--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve - GEO Metastasis Prediction")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(roc_path, dpi=150)
                plt.close()
                print(f"  ROC曲线已保存: {roc_path}")
            else:
                print(f"  警告: y_bin长度({len(y_bin)})与probs长度({len(probs)})不匹配，跳过ROC曲线")
                auc = None
        else:
            print(f"  警告: 模型有{len(classes)}个类别，当前仅支持二分类ROC曲线")
            auc = None
    except Exception as e:
        print(f"  生成ROC曲线时出错: {e}")
        import traceback
        traceback.print_exc()
        auc = None

# 输出结果
print("\n" + "=" * 60)
print("预测结果:")
print("=" * 60)
print(f"预测已保存: {csv_path}")
print(f"准确率: {acc:.4f}")
if auc is not None:
    print(f"AUC: {auc:.4f}")
    if roc_path:
        print(f"ROC图: {roc_path}")

print("\n混淆矩阵:")
print(cm)

print("\n分类报告:")
print(report)

print("\n最佳模型参数:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n原始样本数: {data.shape[0]}")
print(f"X 总行数: {X.shape[0]}")
print(f"X_train 行数: {X_train.shape[0]}")
print(f"X_test  行数: {X_test.shape[0]}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)

