import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

print("=" * 100)
print("TCGA-STAD 和 GEO数据集统计信息汇总")
print("=" * 100)
print()

# 1. TCGA-STAD 数据集分析
print("【TCGA-STAD 数据集】")
print("-" * 100)

TCGA_PROCESSED = BASE_DIR / "data" / "STAD_processed" / "stad_clinical_processed.csv"
tcga_samples = 0
if TCGA_PROCESSED.exists():
    tcga_data = pd.read_csv(TCGA_PROCESSED)
    tcga_samples = len(tcga_data)
    print(f"1. 基本信息:")
    print(f"   - 样本数: {tcga_samples}")
    print(f"   - 特征数: {len(tcga_data.columns) - 2}")
    print(f"   - 数据来源: TCGA (The Cancer Genome Atlas)")
    print(f"   - 平台: Illumina HiSeq")
    
    print(f"\n2. 标签分布:")
    label_counts = tcga_data['label'].value_counts(dropna=False)
    for label, count in label_counts.items():
        if pd.isna(label):
            print(f"   - 未知 (NaN): {count} ({count/len(tcga_data)*100:.1f}%)")
        else:
            print(f"   - {'有转移' if label == 1 else '无转移'} ({label}): {count} ({count/len(tcga_data)*100:.1f}%)")
    
    print(f"\n3. 有效样本:")
    valid_samples = tcga_data.dropna(subset=['label'])
    print(f"   - 有效样本数: {len(valid_samples)} ({len(valid_samples)/len(tcga_data)*100:.1f}%)")
    print(f"   - 无效样本数: {len(tcga_data) - len(valid_samples)} ({(len(tcga_data) - len(valid_samples))/len(tcga_data)*100:.1f}%)")

print()

# 2. GEO-GSE62254 数据集分析
print("【GEO-GSE62254 数据集】")
print("-" * 100)

GSE62254_PATH = BASE_DIR / "data" / "GEO" / "GSE62254_series_matrix.txt"
gse62254_samples = 0
if GSE62254_PATH.exists():
    geo_data = {}
    with open(GSE62254_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = parts[1:] if len(parts) > 1 else []
                geo_data[key] = values
    
    gse62254_samples = len(geo_data.get('!Sample_geo_accession', []))
    print(f"1. 基本信息:")
    print(f"   - 数据集ID: {geo_data.get('!Series_geo_accession', ['N/A'])[0]}")
    print(f"   - 标题: {geo_data.get('!Series_title', ['N/A'])[0]}")
    print(f"   - 平台: {geo_data.get('!Series_platform_id', ['N/A'])[0]}")
    print(f"   - 样本数: {gse62254_samples}")
    print(f"   - 数据来源: GEO (Gene Expression Omnibus)")
    
    print(f"\n2. 数据特点:")
    print(f"   - 包含{gse62254_samples}个胃癌患者的基因表达数据")
    print(f"   - 包含详细的临床随访信息")
    print(f"   - 包含分子分型信息（MSI、Mesenchymal、TP53等）")
    print(f"   - 包含复发和转移信息")

print()

# 3. GEO-GSE26901 数据集分析
print("【GEO-GSE26901 数据集】")
print("-" * 100)

GSE26901_PATH = BASE_DIR / "data" / "GEO" / "GSE26901_GC_KosinUniv_ClinicalInformation.txt"
gse26901_samples = 0
if GSE26901_PATH.exists():
    gse26901_data = pd.read_csv(GSE26901_PATH, sep='\t')
    gse26901_samples = len(gse26901_data)
    
    print(f"1. 基本信息:")
    print(f"   - 样本数: {gse26901_samples}")
    print(f"   - 特征数: {len(gse26901_data.columns)}")
    print(f"   - 数据来源: GEO (Gene Expression Omnibus)")
    print(f"   - 机构: 高丽大学")
    
    print(f"\n2. 性别分布:")
    gender_counts = gse26901_data['Gender'].value_counts()
    for gender, count in gender_counts.items():
        print(f"   - {gender}: {count} ({count/len(gse26901_data)*100:.1f}%)")
    
    print(f"\n3. 年龄统计:")
    ages = gse26901_data['Age'].dropna()
    if len(ages) > 0:
        print(f"   - 平均值: {np.mean(ages):.1f} ± {np.std(ages):.1f} 岁")
        print(f"   - 范围: {min(ages):.0f} - {max(ages):.0f} 岁")
        print(f"   - 有效样本数: {len(ages)}")
    
    print(f"\n4. Lauren分型分布:")
    lauren_counts = gse26901_data['Lauren'].value_counts()
    for lauren, count in lauren_counts.items():
        if pd.notna(lauren):
            print(f"   - {lauren}: {count} ({count/len(gse26901_data)*100:.1f}%)")
    
    print(f"\n5. AJCC分期分布:")
    stage_counts = gse26901_data['AJCC Stage'].value_counts()
    for stage, count in stage_counts.items():
        print(f"   - Stage {stage}: {count} ({count/len(gse26901_data)*100:.1f}%)")

print()

# 4. GEO-GSE84437 数据集分析
print("【GEO-GSE84437 数据集】")
print("-" * 100)

GSE84437_PATH = BASE_DIR / "data" / "GEO" / "GSE84437_series_matrix.txt"
gse84437_samples = 0
if GSE84437_PATH.exists():
    geo_data = {}
    with open(GSE84437_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = parts[1:] if len(parts) > 1 else []
                geo_data[key] = values
    
    gse84437_samples = len(geo_data.get('!Sample_geo_accession', []))
    print(f"1. 基本信息:")
    print(f"   - 数据集ID: {geo_data.get('!Series_geo_accession', ['N/A'])[0]}")
    print(f"   - 标题: {geo_data.get('!Series_title', ['N/A'])[0]}")
    print(f"   - 平台: {geo_data.get('!Series_platform_id', ['N/A'])[0]}")
    print(f"   - 样本数: {gse84437_samples}")
    print(f"   - 数据来源: GEO (Gene Expression Omnibus)")
    print(f"   - 机构: 延世大学")
    
    print(f"\n2. 数据特点:")
    print(f"   - 专注于胃癌分子分型研究")
    print(f"   - 包含多个胃癌分子亚型")
    print(f"   - 包含基因表达数据和临床信息")

print()

# 5. 数据集对比
print("【数据集对比】")
print("-" * 100)
print(f"{'项目':<30} {'TCGA-STAD':<20} {'GSE62254':<15} {'GSE26901':<15} {'GSE84437':<15}")
print("-" * 100)

# 数据来源
print(f"{'数据来源':<30} {'TCGA':<20} {'GEO':<15} {'GEO':<15} {'GEO':<15}")

# 样本数量
print(f"{'样本数量':<30} {tcga_samples:<20} {gse62254_samples:<15} {gse26901_samples:<15} {gse84437_samples:<15}")

# 数据类型
print(f"{'数据类型':<30} {'临床+基因表达':<20} {'基因表达+临床':<15} {'临床信息':<15} {'基因表达+临床':<15}")

# 平台
print(f"{'平台':<30} {'Illumina HiSeq':<20} {'Affymetrix':<15} {'Affymetrix':<15} {'Illumina':<15}")

# 种族分布
print(f"{'种族分布':<30} {'多种族':<20} {'亚洲人群':<15} {'亚洲人群':<15} {'亚洲人群':<15}")

# 转移标签
print(f"{'转移标签':<30} {'M分期+肿瘤分期':<20} {'临床随访':<15} {'AJCC分期':<15} {'分子分型':<15}")

# 有效样本率
if TCGA_PROCESSED.exists():
    valid_rate = len(tcga_data.dropna(subset=['label'])) / len(tcga_data) * 100
    print(f"{'有效样本率':<30} {f'{valid_rate:.1f}%':<20} {'需要分析':<15} {'需要分析':<15} {'需要分析':<15}")

print()

# 6. GEO和TCGA数据整合方法
print("【GEO和TCGA数据整合方法】")
print("-" * 100)

print("1. 数据格式统一:")
print("   - 统一基因标识符（如Entrez ID、Gene Symbol）")
print("   - 统一特征命名规范")
print("   - 统一数据类型和单位")

print("\n2. 数据标准化:")
print("   - 对基因表达数据进行标准化（如z-score标准化）")
print("   - 对临床特征进行归一化处理")
print("   - 确保不同数据集的分布一致")

print("\n3. 批次效应处理:")
print("   - 使用ComBat或SVA方法处理批次效应")
print("   - 识别并校正平台差异")
print("   - 验证处理后的数据分布")

print("\n4. 特征选择:")
print("   - 选择在多个数据集中都存在的基因/特征")
print("   - 进行特征重要性分析")
print("   - 选择与转移相关的关键特征")

print("\n5. 标签对齐:")
print("   - 统一转移标签的定义（如M1、复发等）")
print("   - 确保标签的一致性")
print("   - 处理缺失标签的样本")

print("\n6. 模型训练策略:")
print("   - 方案A: 在TCGA上训练，在GEO上验证")
print("   - 方案B: 在GEO上训练，在TCGA上验证")
print("   - 方案C: 合并数据集进行训练，使用交叉验证")
print("   - 方案D: 使用迁移学习方法")

print("\n7. 模型验证:")
print("   - 在独立数据集上验证模型性能")
print("   - 使用AUC、准确率、F1分数等指标")
print("   - 进行敏感性分析")
print("   - 评估模型的泛化能力")

print()
print("=" * 100)
print("分析完成")
print("=" * 100)
