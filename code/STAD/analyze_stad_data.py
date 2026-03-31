import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CLINICAL_PATH = BASE_DIR / "data" / "STAD" / "clinical.project-tcga-stad.2026-01-25" / "clinical.tsv"
PROCESSED_PATH = BASE_DIR / "data" / "STAD_processed" / "stad_clinical_processed.csv"

print("=== STAD数据统计分析 ===")
print()

# 1. 原始数据统计
print("1. 原始临床数据统计:")
if CLINICAL_PATH.exists():
    clinical = pd.read_csv(CLINICAL_PATH, sep="\t", low_memory=False)
    print(f"  原始记录数: {len(clinical)}")
    print(f"  唯一病例数: {clinical['cases.submitter_id'].nunique()}")
    print(f"  数据列数: {len(clinical.columns)}")
else:
    print("  临床数据文件不存在")

print()

# 2. 处理后数据统计
print("2. 处理后数据统计:")
if PROCESSED_PATH.exists():
    processed = pd.read_csv(PROCESSED_PATH)
    print(f"  处理后样本数: {len(processed)}")
    print(f"  特征数: {len(processed.columns) - 2}")  # 减去sample_id和label
    
    # 标签分布
    label_counts = processed['label'].value_counts(dropna=False)
    print(f"  标签分布:")
    print(f"    有转移 (1): {label_counts.get(1.0, 0)}")
    print(f"    无转移 (0): {label_counts.get(0.0, 0)}")
    print(f"    未知 (NaN): {label_counts.get(np.nan, 0)}")
    
    # 有效样本数
    valid_samples = processed.dropna(subset=['label'])
    print(f"  有效样本数: {len(valid_samples)}")
    print(f"  无效样本数: {len(processed) - len(valid_samples)}")
else:
    print("  处理后数据文件不存在")

print()

# 3. 数据处理流程
print("3. 数据处理流程:")
print("  1. 数据清洗:")
print("     - 处理缺失值（将'--'、'not reported'等转换为NaN）")
print("     - 标准化数据格式")
print("  2. 标签提取:")
print("     - 从多个字段提取转移标签:")
print("       * diagnoses.metastasis_at_diagnosis")
print("       * diagnoses.ajcc_pathologic_m")
print("       * diagnoses.ajcc_clinical_m")
print("       * diagnoses.uicc_pathologic_m")
print("       * 肿瘤分期（IV期视为有转移）")
print("  3. 特征处理:")
print("     - 按病例ID聚合数据")
print("     - 合并样本特征")
print("     - 删除缺失率高于60%的特征")
print("  4. 模型训练:")
print("     - 数据编码（对分类变量进行因子化）")
print("     - 训练集和测试集划分（80%/20%）")
print("     - 随机森林分类器，使用5折交叉验证")
print("     - 超参数搜索优化模型")

print()

# 4. GEO和TCGA数据整合建议
print("4. GEO和TCGA数据整合建议:")
print("  1. 数据格式统一:")
print("     - 确保两个数据集的特征名称一致")
print("     - 统一数据类型和单位")
print("  2. 数据标准化:")
print("     - 对数值特征进行标准化处理")
print("     - 对分类特征进行一致的编码")
print("  3. 批次效应处理:")
print("     - 使用ComBat等方法处理批次效应")
print("     - 确保不同来源的数据分布一致")
print("  4. 特征选择:")
print("     - 选择在两个数据集中都存在的特征")
print("     - 进行特征重要性分析，选择关键特征")
print("  5. 模型验证:")
print("     - 在TCGA数据上训练模型")
print("     - 在GEO数据上验证模型性能")
print("     - 进行交叉验证确保模型泛化能力")

print()

# 5. 样本统计详情
if PROCESSED_PATH.exists():
    print("5. 样本统计详情:")
    # 年龄统计
    age_col = 'demographic.age_at_index'
    if age_col in processed.columns:
        age_valid = processed[age_col].dropna()
        print(f"  年龄统计:")
        print(f"    平均值: {age_valid.mean():.1f}")
        print(f"    标准差: {age_valid.std():.1f}")
        print(f"    范围: {age_valid.min():.1f} - {age_valid.max():.1f}")
    
    # 性别分布
    gender_col = 'demographic.gender'
    if gender_col in processed.columns:
        gender_counts = processed[gender_col].value_counts()
        print(f"  性别分布:")
        for gender, count in gender_counts.items():
            print(f"    {gender}: {count}")

print()
print("=== 分析完成 ===")
