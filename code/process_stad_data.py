"""
TCGA-STAD数据清洗脚本
从TCGA-STAD的临床和生物样本数据中提取可用于预测转移的特征和标签
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 设置路径
BASE_DIR = Path(__file__).resolve().parents[1]
STAD_DIR = BASE_DIR / "data" / "STAD"
CLINICAL_DIR = STAD_DIR / "clinical.project-tcga-stad.2026-01-25"
BIOSPECIMEN_DIR = STAD_DIR / "biospecimen.project-tcga-stad.2026-01-25"
OUTPUT_DIR = BASE_DIR / "data" / "STAD"
OUTPUT_CSV = OUTPUT_DIR / "stad_processed.csv"

print("=" * 60)
print("TCGA-STAD数据清洗脚本")
print("=" * 60)

# 1. 读取临床数据
print("\n[1/4] 读取临床数据...")
clinical_path = CLINICAL_DIR / "clinical.tsv"
if not clinical_path.exists():
    raise FileNotFoundError(f"临床数据文件不存在: {clinical_path}")

clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
print(f"  读取临床数据: {clinical.shape[0]} 行, {clinical.shape[1]} 列")

# 2. 读取样本数据（用于关联样本类型）
print("\n[2/4] 读取生物样本数据...")
sample_path = BIOSPECIMEN_DIR / "sample.tsv"
if not sample_path.exists():
    raise FileNotFoundError(f"样本数据文件不存在: {sample_path}")

samples = pd.read_csv(sample_path, sep="\t", low_memory=False)
print(f"  读取样本数据: {samples.shape[0]} 行, {samples.shape[1]} 列")

# 3. 提取关键特征和标签
print("\n[3/4] 提取关键特征和标签...")

# 3.1 提取转移状态（主要标签）
# 使用多个字段来确定转移状态
def extract_metastasis_status(row):
    """从多个字段提取转移状态"""
    # 方法1: 直接转移字段
    if pd.notna(row.get('diagnoses.metastasis_at_diagnosis')):
        meta_val = str(row['diagnoses.metastasis_at_diagnosis']).lower()
        if 'metastasis' in meta_val or meta_val == 'yes':
            return 1
        elif meta_val in ['no', 'none', '']:
            return 0
    
    # 方法2: AJCC M分期 (M0=无转移, M1=有转移, MX=未知)
    if pd.notna(row.get('diagnoses.ajcc_pathologic_m')):
        m_stage = str(row['diagnoses.ajcc_pathologic_m']).upper()
        if m_stage == 'M1':
            return 1
        elif m_stage == 'M0':
            return 0
    
    # 方法3: 临床M分期
    if pd.notna(row.get('diagnoses.ajcc_clinical_m')):
        m_stage = str(row['diagnoses.ajcc_clinical_m']).upper()
        if m_stage == 'M1':
            return 1
        elif m_stage == 'M0':
            return 0
    
    # 方法4: UICC M分期
    if pd.notna(row.get('diagnoses.uicc_pathologic_m')):
        m_stage = str(row['diagnoses.uicc_pathologic_m']).upper()
        if m_stage == 'M1':
            return 1
        elif m_stage == 'M0':
            return 0
    
    # 如果都找不到，返回NaN
    return np.nan

# 应用转移状态提取
clinical['metastasis_label'] = clinical.apply(extract_metastasis_status, axis=1)

# 3.2 选择每个病例的主要诊断记录
# 先检查字段是否存在以及值的分布
if 'diagnoses.diagnosis_is_primary_disease' in clinical.columns:
    print(f"  检查diagnosis_is_primary_disease字段值分布:")
    value_counts = clinical['diagnoses.diagnosis_is_primary_disease'].value_counts()
    print(f"    唯一值: {list(value_counts.index)}")
    print(f"    值分布: {value_counts.to_dict()}")
    print(f"    缺失值: {clinical['diagnoses.diagnosis_is_primary_disease'].isna().sum()}")
    
    # 尝试多种可能的真值表示
    primary_mask = (
        (clinical['diagnoses.diagnosis_is_primary_disease'] == 'true') |
        (clinical['diagnoses.diagnosis_is_primary_disease'] == 'True') |
        (clinical['diagnoses.diagnosis_is_primary_disease'] == True) |
        (clinical['diagnoses.diagnosis_is_primary_disease'] == 1) |
        (clinical['diagnoses.diagnosis_is_primary_disease'] == '1') |
        (clinical['diagnoses.diagnosis_is_primary_disease'].isna())
    )
    
    # 如果筛选后没有数据，使用所有记录
    if primary_mask.sum() == 0:
        print("  警告: 筛选条件未匹配到任何记录，使用所有记录")
        primary_diagnoses = clinical.copy()
    else:
        primary_diagnoses = clinical[primary_mask].copy()
        print(f"  筛选后记录数: {len(primary_diagnoses)}")
else:
    print("  警告: diagnoses.diagnosis_is_primary_disease 字段不存在，使用所有记录")
    primary_diagnoses = clinical.copy()

# 如果有多条记录，选择第一条（按case_id分组）
if 'cases.case_id' in primary_diagnoses.columns:
    primary_diagnoses = primary_diagnoses.groupby('cases.case_id').first().reset_index()
else:
    print("  警告: cases.case_id 字段不存在，使用所有记录")
    # 如果没有case_id，尝试使用submitter_id
    if 'cases.submitter_id' in primary_diagnoses.columns:
        primary_diagnoses = primary_diagnoses.groupby('cases.submitter_id').first().reset_index()

print(f"  提取主要诊断记录: {primary_diagnoses.shape[0]} 个病例")

# 3.3 提取临床特征
feature_cols = {
    # 人口统计学特征
    'age': 'demographic.age_at_index',
    'gender': 'demographic.gender',
    'vital_status': 'demographic.vital_status',
    'days_to_death': 'demographic.days_to_death',
    
    # 诊断特征
    'age_at_diagnosis': 'diagnoses.age_at_diagnosis',
    'tumor_grade': 'diagnoses.tumor_grade',
    'tumor_stage': 'diagnoses.ajcc_pathologic_stage',
    't_stage': 'diagnoses.ajcc_pathologic_t',
    'n_stage': 'diagnoses.ajcc_pathologic_n',
    'm_stage': 'diagnoses.ajcc_pathologic_m',
    
    # 其他重要特征
    'primary_diagnosis': 'diagnoses.primary_diagnosis',
    'morphology': 'diagnoses.morphology',
}

# 创建特征数据框
features_dict = {}
for new_name, old_name in feature_cols.items():
    if old_name in primary_diagnoses.columns:
        features_dict[new_name] = primary_diagnoses[old_name]
    else:
        print(f"  警告: 列 {old_name} 不存在，跳过")

# 添加病例ID和转移标签
features_dict['case_id'] = primary_diagnoses['cases.case_id']
features_dict['submitter_id'] = primary_diagnoses['cases.submitter_id']
features_dict['label'] = primary_diagnoses['metastasis_label']

# 创建特征数据框
features_df = pd.DataFrame(features_dict)

# 3.4 处理分类变量
print("\n[4/4] 处理分类变量和缺失值...")

# 性别编码
if 'gender' in features_df.columns:
    features_df['gender'] = features_df['gender'].map({
        'male': 1, 'female': 0, 'Male': 1, 'Female': 0
    }).fillna(-1)

# 生存状态编码
if 'vital_status' in features_df.columns:
    features_df['vital_status'] = features_df['vital_status'].map({
        'Dead': 1, 'Alive': 0, 'dead': 1, 'alive': 0
    }).fillna(-1)

# T/N分期编码（简化：T1-T2=0, T3-T4=1, N0=0, N1-N3=1）
if 't_stage' in features_df.columns:
    features_df['t_stage_encoded'] = features_df['t_stage'].apply(
        lambda x: 1 if pd.notna(x) and ('T3' in str(x) or 'T4' in str(x)) else 
                  (0 if pd.notna(x) and ('T1' in str(x) or 'T2' in str(x)) else -1)
    )

if 'n_stage' in features_df.columns:
    features_df['n_stage_encoded'] = features_df['n_stage'].apply(
        lambda x: 1 if pd.notna(x) and ('N1' in str(x) or 'N2' in str(x) or 'N3' in str(x)) else 
                  (0 if pd.notna(x) and 'N0' in str(x) else -1)
    )

# 删除原始的分期列（如果已编码）
if 't_stage_encoded' in features_df.columns:
    features_df = features_df.drop(columns=['t_stage', 'n_stage', 'm_stage'], errors='ignore')

# 3.5 处理缺失值和数值列
# 将数值列转换为数值类型
numeric_cols = ['age', 'age_at_diagnosis', 'days_to_death']
for col in numeric_cols:
    if col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

# 删除缺失值过多的行（标签缺失或关键特征缺失超过50%）
label_col = 'label'
key_features = ['age', 'age_at_diagnosis', 'gender']
missing_threshold = len(key_features) * 0.5

before_count = len(features_df)
# 删除标签缺失的行
features_df = features_df[features_df[label_col].notna()].copy()
# 删除关键特征缺失过多的行
if key_features:
    available_key_features = [f for f in key_features if f in features_df.columns]
    if available_key_features:
        missing_count = features_df[available_key_features].isna().sum(axis=1)
        features_df = features_df[missing_count <= missing_threshold].copy()

after_count = len(features_df)
print(f"  删除缺失值过多的行: {before_count} -> {after_count}")

# 3.6 填充剩余缺失值（数值列用中位数，分类列用众数）
for col in features_df.columns:
    if col in ['case_id', 'submitter_id', 'label']:
        continue
    if features_df[col].dtype in ['int64', 'float64']:
        features_df[col] = features_df[col].fillna(features_df[col].median())
    else:
        features_df[col] = features_df[col].fillna(features_df[col].mode()[0] if len(features_df[col].mode()) > 0 else -1)

# 3.7 设置索引
features_df = features_df.set_index('submitter_id')

# 4. 保存处理后的数据
print(f"\n保存处理后的数据到: {OUTPUT_CSV}")
features_df.to_csv(OUTPUT_CSV)
print(f"  保存完成: {features_df.shape[0]} 个样本, {features_df.shape[1]} 个特征")

# 5. 数据统计
print("\n" + "=" * 60)
print("数据统计:")
print("=" * 60)
print(f"总样本数: {len(features_df)}")

if len(features_df) == 0:
    print("\n警告: 没有提取到任何数据！")
    print("可能的原因:")
    print("  1. 数据文件格式不正确")
    print("  2. 筛选条件太严格")
    print("  3. 字段名称不匹配")
    print("\n请检查:")
    print("  - 临床数据文件路径是否正确")
    print("  - 数据文件是否包含预期的列名")
    sys.exit(1)

if 'label' in features_df.columns:
    label_counts = features_df['label'].value_counts()
    print(f"\n转移状态分布:")
    print(f"  无转移 (0): {label_counts.get(0, 0)}")
    print(f"  有转移 (1): {label_counts.get(1, 0)}")
    missing_label = features_df['label'].isna().sum()
    if missing_label > 0:
        print(f"  缺失值: {missing_label}")
    
    if len(features_df) > 0:
        transfer_rate = label_counts.get(1, 0) / len(features_df) * 100
        print(f"  转移率: {transfer_rate:.2f}%")
    else:
        print("  转移率: N/A (无数据)")

print(f"\n特征列: {list(features_df.columns)}")
print(f"\n前5行数据:")
print(features_df.head().to_string())

print("\n" + "=" * 60)
print("注意: 此数据集仅包含临床特征，不包含基因表达数据。")
print("如需进行基因表达预测，请从TCGA下载RNA-seq或microarray数据。")
print("=" * 60)



