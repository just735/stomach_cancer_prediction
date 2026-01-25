"""
GEO数据清洗脚本
从GEO数据集中提取可用于预测转移的特征和标签
参考 process_stad_data.py 的结构
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# 设置路径
BASE_DIR = Path(__file__).resolve().parents[1]
GEO_DIR = BASE_DIR / "data" / "GEO"
OUTPUT_DIR = BASE_DIR / "data" / "GEO"
OUTPUT_CSV = OUTPUT_DIR / "geo_processed.csv"

print("=" * 60)
print("GEO数据清洗脚本")
print("=" * 60)

# 1. 读取基因表达数据
print("\n[1/4] 读取基因表达数据...")
expr_path = GEO_DIR / "GSE15459_series_matrix.txt"
if not expr_path.exists():
    raise FileNotFoundError(f"表达数据文件不存在: {expr_path}")

try:
    # 读取表达矩阵（跳过以!开头的注释行）
    expr = pd.read_csv(expr_path, sep="\t", comment="!", index_col=0, engine="python", on_bad_lines="skip")
    print(f"  读取表达数据: {expr.shape[0]} 个基因, {expr.shape[1]} 个样本")
    
    # 转置：行→样本，列→基因
    expr_t = expr.T.reset_index().rename(columns={"index": "sample_id"})
    print(f"  转置后: {expr_t.shape[0]} 个样本, {expr_t.shape[1]} 个基因特征")
except Exception as e:
    raise FileNotFoundError(f"读取表达矩阵失败: {e}")

# 2. 读取临床数据
print("\n[2/4] 读取临床数据...")
clin_path = GEO_DIR / "GSE15459_outcome.xls"
if not clin_path.exists():
    # 尝试其他可能的文件名
    alt_paths = [
        GEO_DIR / "GSE15459_outcome.xlsx",
        GEO_DIR / "outcome.xls",
        GEO_DIR / "outcome.xlsx"
    ]
    clin_path = None
    for p in alt_paths:
        if p.exists():
            clin_path = p
            break
    
    if clin_path is None:
        raise FileNotFoundError(f"临床数据文件不存在，请检查 {GEO_DIR} 目录")

try:
    # 根据文件后缀选择读取方式
    suf = clin_path.suffix.lower()
    if suf == ".xls":
        try:
            clinical = pd.read_excel(clin_path, engine="xlrd")
        except ImportError:
            print("  错误: 缺少xlrd模块，无法读取.xls文件")
            print("  请运行: pip install xlrd")
            print("  或者将文件转换为.xlsx格式")
            raise
    elif suf == ".xlsx":
        try:
            clinical = pd.read_excel(clin_path, engine="openpyxl")
        except ImportError:
            print("  错误: 缺少openpyxl模块，无法读取.xlsx文件")
            print("  请运行: pip install openpyxl")
            raise
    else:
        # 尝试作为CSV/TSV文件读取
        clinical = pd.read_csv(clin_path, sep="\t", engine="python", on_bad_lines="skip")
    print(f"  读取临床数据: {clinical.shape[0]} 行, {clinical.shape[1]} 列")
except ImportError as e:
    # 重新抛出ImportError，让用户知道需要安装什么
    raise
except Exception as e:
    print(f"  错误: 读取临床信息失败: {e}")
    print("  可能的解决方案:")
    print("    1. 对于.xls文件: pip install xlrd")
    print("    2. 对于.xlsx文件: pip install openpyxl")
    print("    3. 或将文件转换为CSV格式")
    raise FileNotFoundError(f"读取临床信息失败: {e}")

# 3. 提取关键特征和标签
print("\n[3/4] 提取关键特征和标签...")

# 3.1 找到样本ID列
possible = [c for c in clinical.columns if "GSM" in str(c) or "sample" in str(c).lower()]
if not possible:
    raise ValueError("临床表中找不到包含 GSM 的列，请确认列名。")
gsm_col = possible[0]
print(f"  使用样本ID列: {gsm_col}")

# 3.2 排除已知不合格样本（如果存在）
excluded = ["GSM387788", "GSM387790", "GSM387793", "GSM387797",
            "GSM387798", "GSM387799", "GSM387842"]
expr_clean = expr_t[~expr_t["sample_id"].isin(excluded)].copy()
if len(expr_clean) < len(expr_t):
    print(f"  排除不合格样本: {len(expr_t) - len(expr_clean)} 个")

# 3.3 提取转移状态标签
def extract_metastasis_label(row):
    """从多个字段提取转移状态"""
    # 方法1: 直接查找转移相关字段
    metastasis_cols = [c for c in clinical.columns if "metastasis" in str(c).lower()]
    for col in metastasis_cols:
        if pd.notna(row.get(col)):
            val = str(row[col]).lower()
            if any(x in val for x in ["yes", "1", "metastasis", "positive", "true"]):
                return 1
            elif any(x in val for x in ["no", "0", "none", "negative", "false"]):
                return 0
    
    # 方法2: 从Outcome字段推断（如果Outcome表示死亡，可能暗示转移）
    outcome_cols = [c for c in clinical.columns if "outcome" in str(c).lower()]
    for col in outcome_cols:
        if pd.notna(row.get(col)):
            val = str(row[col]).lower()
            # 如果Outcome=1表示死亡，可能暗示转移
            if "1" in val or "dead" in val:
                return 1
            elif "0" in val or "alive" in val:
                return 0
    
    # 方法3: 从Stage字段推断（晚期可能暗示转移）
    stage_cols = [c for c in clinical.columns if "stage" in str(c).lower()]
    for col in stage_cols:
        if pd.notna(row.get(col)):
            val = str(row[col]).upper()
            # Stage IV通常表示转移
            if "IV" in val or "4" in val:
                return 1
    
    return np.nan

# 应用转移状态提取
clinical['metastasis_label'] = clinical.apply(extract_metastasis_label, axis=1)

# 3.4 选择核心临床列
cols_to_pick = [gsm_col, "Age_at_surgery", "Gender", "Laurenclassification",
                "Stage", "Overall.Survival (Months)**", "Outcome (1=dead)"]
exist_cols = [c for c in cols_to_pick if c in clinical.columns]
clinical_core = clinical[exist_cols + ['metastasis_label']].rename(columns={gsm_col: "sample_id"})

print(f"  提取的临床特征: {len(exist_cols)} 个")
print(f"  转移标签提取情况:")
label_counts = clinical_core['metastasis_label'].value_counts(dropna=False)
print(f"    有转移 (1): {label_counts.get(1, 0)}")
print(f"    无转移 (0): {label_counts.get(0, 0)}")
print(f"    缺失值: {label_counts.get(np.nan, 0) if pd.isna(label_counts.index).any() else 0}")

# 3.5 合并表达数据和临床数据
print("\n[4/4] 合并数据并处理...")
merged = pd.merge(expr_clean, clinical_core, on="sample_id", how="inner")
if merged.shape[0] == 0:
    raise ValueError("合并后无样本，检查 sample_id/GSM 是否匹配。")

print(f"  合并后样本数: {merged.shape[0]}")

# 4. 处理分类变量和缺失值
# 4.1 对分类列做编码
le_dict = {}
for col in ["Gender", "Laurenclassification"]:
    if col in merged.columns:
        # 处理缺失值
        merged[col] = merged[col].fillna("NA").astype(str)
        le = LabelEncoder()
        merged[col] = le.fit_transform(merged[col])
        le_dict[col] = le
        print(f"  编码列 {col}: {len(le.classes_)} 个类别")
    else:
        merged[col] = 0
        print(f"  警告: 列 {col} 不存在，填充为0")

# 4.2 处理Stage列（如果有）
if "Stage" in merged.columns:
    # 将Stage编码为数值
    stage_mapping = {
        'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
        'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
        'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
        'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4
    }
    merged['Stage'] = merged['Stage'].fillna("Unknown")
    merged['Stage_encoded'] = merged['Stage'].map(
        lambda x: stage_mapping.get(str(x), 0) if str(x) in stage_mapping else 0
    )
    merged = merged.drop(columns=['Stage'])
    print(f"  编码Stage列")

# 4.3 处理缺失值
# 删除转移标签缺失的行
before_count = len(merged)
merged = merged[merged['metastasis_label'].notna()].copy()
if len(merged) < before_count:
    print(f"  删除转移标签缺失的行: {before_count - len(merged)} 个")

# 删除缺失值过多的行（缺失值超过50%）
missing_threshold = merged.shape[1] * 0.5
before_count = len(merged)
merged = merged[merged.isna().sum(axis=1) < missing_threshold].copy()
if len(merged) < before_count:
    print(f"  删除缺失值过多的行: {before_count - len(merged)} 个")

# 用中位数填充数值列的缺失值
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged[col].isna().sum() > 0:
        merged[col] = merged[col].fillna(merged[col].median())

# 4.4 将sample_id设为索引，标签放到最后
merged = merged.set_index("sample_id", drop=True)
if 'metastasis_label' in merged.columns:
    cols = [c for c in merged.columns if c != 'metastasis_label'] + ['metastasis_label']
    merged = merged[cols]
    merged = merged.rename(columns={'metastasis_label': 'label'})

# 5. 保存处理后的数据
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_CSV, index=True)
print(f"\n保存处理后的数据到: {OUTPUT_CSV}")
print(f"  保存完成: {merged.shape[0]} 个样本, {merged.shape[1]} 个特征")

# 6. 数据统计
print("\n" + "=" * 60)
print("数据统计:")
print("=" * 60)
print(f"总样本数: {len(merged)}")

if 'label' in merged.columns:
    label_counts = merged['label'].value_counts()
    print(f"\n转移状态分布:")
    print(f"  无转移 (0): {label_counts.get(0, 0)}")
    print(f"  有转移 (1): {label_counts.get(1, 0)}")
    if len(merged) > 0:
        transfer_rate = label_counts.get(1, 0) / len(merged) * 100
        print(f"  转移率: {transfer_rate:.2f}%")

print(f"\n特征列数: {merged.shape[1] - 1} (不包括标签列)")
print(f"基因特征数: {merged.shape[1] - len(exist_cols) - 1} (不包括临床特征和标签)")

print("\n前5行数据预览:")
print(merged.head().to_string())

print("\n" + "=" * 60)
print("注意: 此数据集包含基因表达数据和临床特征。")
print("=" * 60)

