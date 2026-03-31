import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GEO_PATH = BASE_DIR / "data" / "GEO" / "GSE62254_series_matrix.txt"

print("=" * 100)
print("GEO-GSE62254 数据集详细分析")
print("=" * 100)
print()

# 读取GEO数据并提取样本特征
geo_data = {}
sample_info = {}

with open(GEO_PATH, 'r', encoding='utf-8') as f:
    current_key = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('!'):
            parts = line.split('\t')
            key = parts[0]
            values = parts[1:] if len(parts) > 1 else []
            geo_data[key] = values
            
            # 提取样本ID
            if key == '!Sample_geo_accession':
                for i, sample_id in enumerate(values):
                    if sample_id not in sample_info:
                        sample_info[sample_id] = {}
                    sample_info[sample_id]['geo_accession'] = sample_id

# 提取样本特征
sample_chars = geo_data.get('!Sample_characteristics_ch1', [])
sample_titles = geo_data.get('!Sample_title', [])
sample_accessions = geo_data.get('!Sample_geo_accession', [])

print("【样本基本信息】")
print("-" * 100)
print(f"总样本数: {len(sample_accessions)}")
print(f"样本ID范围: {sample_accessions[0]} 到 {sample_accessions[-1]}")
print()

# 提取所有样本的特征
all_features = set()
for char in sample_chars:
    if isinstance(char, str):
        features = [f.split(':')[0].strip() for f in char.split(';') if ':' in f]
        all_features.update(features)

print(f"发现 {len(all_features)} 个临床特征:")
for i, feature in enumerate(sorted(all_features), 1):
    print(f"  {i:2d}. {feature}")

print()

# 分析每个特征的数据分布
print("【特征数据分布】")
print("-" * 100)

feature_data = {}
for i, (char, title, accession) in enumerate(zip(sample_chars, sample_titles, sample_accessions)):
    if isinstance(char, str):
        features = char.split(';')
        for feature in features:
            if ':' in feature:
                key, value = feature.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key not in feature_data:
                    feature_data[key] = {}
                
                if accession not in feature_data[key]:
                    feature_data[key][accession] = []
                feature_data[key][accession].append(value)

# 统计每个特征的数据完整性
print(f"{'特征名称':<30} {'有效样本数':<15} {'缺失样本数':<15} {'完整率':<10} {'唯一值数量':<15}")
print("-" * 100)

for feature in sorted(all_features):
    if feature in feature_data:
        valid_count = len(feature_data[feature])
        missing_count = len(sample_accessions) - valid_count
        completeness = valid_count / len(sample_accessions) * 100
        
        # 统计唯一值数量
        all_values = []
        for accession in feature_data[feature]:
            all_values.extend(feature_data[feature][accession])
        unique_values = len(set(all_values))
        
        print(f"{feature:<30} {valid_count:<15} {missing_count:<15} {completeness:>6.1f}% {unique_values:<15}")

print()

# 分析关键特征
print("【关键特征详细分析】")
print("-" * 100)

# 1. 患者ID
if 'patient' in feature_data:
    patient_ids = []
    for accession in feature_data['patient']:
        patient_ids.extend(feature_data['patient'][accession])
    print(f"1. 患者ID (patient):")
    print(f"   - 唯一患者数: {len(set(patient_ids))}")
    print(f"   - 患者ID范围: {min([int(p.split()[-1]) for p in patient_ids if p.split()[-1].isdigit()])} - {max([int(p.split()[-1]) for p in patient_ids if p.split()[-1].isdigit()])}")

# 2. 年龄
if 'age' in feature_data:
    ages = []
    for accession in feature_data['age']:
        ages.extend(feature_data['age'][accession])
    ages = [int(a.split()[0]) for a in ages if a.split()[0].isdigit()]
    if ages:
        print(f"\n2. 年龄 (age):")
        print(f"   - 平均值: {np.mean(ages):.1f} ± {np.std(ages):.1f} 岁")
        print(f"   - 范围: {min(ages)} - {max(ages)} 岁")
        print(f"   - 有效样本数: {len(ages)}")

# 3. 性别
if 'gender' in feature_data:
    genders = []
    for accession in feature_data['gender']:
        genders.extend(feature_data['gender'][accession])
    gender_counts = pd.Series(genders).value_counts()
    print(f"\n3. 性别 (gender):")
    for gender, count in gender_counts.items():
        print(f"   - {gender}: {count} ({count/len(genders)*100:.1f}%)")

# 4. 肿瘤分期
if 'stage' in feature_data:
    stages = []
    for accession in feature_data['stage']:
        stages.extend(feature_data['stage'][accession])
    stage_counts = pd.Series(stages).value_counts()
    print(f"\n4. 肿瘤分期 (stage):")
    for stage, count in stage_counts.items():
        print(f"   - {stage}: {count} ({count/len(stages)*100:.1f}%)")

# 5. 转移状态
if 'metastasis' in feature_data:
    metastasis = []
    for accession in feature_data['metastasis']:
        metastasis.extend(feature_data['metastasis'][accession])
    meta_counts = pd.Series(metastasis).value_counts()
    print(f"\n5. 转移状态 (metastasis):")
    for status, count in meta_counts.items():
        print(f"   - {status}: {count} ({count/len(metastasis)*100:.1f}%)")

# 6. 复发状态
if 'recurrence' in feature_data:
    recurrence = []
    for accession in feature_data['recurrence']:
        recurrence.extend(feature_data['recurrence'][accession])
    recur_counts = pd.Series(recurrence).value_counts()
    print(f"\n6. 复发状态 (recurrence):")
    for status, count in recur_counts.items():
        print(f"   - {status}: {count} ({count/len(recurrence)*100:.1f}%)")

# 7. 分子分型
if 'subtype' in feature_data or 'molecular subtype' in feature_data:
    subtype_key = 'subtype' if 'subtype' in feature_data else 'molecular subtype'
    subtypes = []
    for accession in feature_data[subtype_key]:
        subtypes.extend(feature_data[subtype_key][accession])
    subtype_counts = pd.Series(subtypes).value_counts()
    print(f"\n7. 分子分型 ({subtype_key}):")
    for subtype, count in subtype_counts.items():
        print(f"   - {subtype}: {count} ({count/len(subtypes)*100:.1f}%)")

# 8. 存活状态
if 'vital status' in feature_data or 'status' in feature_data:
    status_key = 'vital status' if 'vital status' in feature_data else 'status'
    statuses = []
    for accession in feature_data[status_key]:
        statuses.extend(feature_data[status_key][accession])
    status_counts = pd.Series(statuses).value_counts()
    print(f"\n8. 存活状态 ({status_key}):")
    for status, count in status_counts.items():
        print(f"   - {status}: {count} ({count/len(statuses)*100:.1f}%)")

print()
print("=" * 100)
print("分析完成")
print("=" * 100)
