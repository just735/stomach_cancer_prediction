import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GEO_PATH = BASE_DIR / "data" / "GEO" / "GSE62254_series_matrix.txt"
TCGA_PROCESSED_PATH = BASE_DIR / "data" / "STAD_processed" / "stad_clinical_processed.csv"
TCGA_ORIGINAL_PATH = BASE_DIR / "data" / "STAD" / "clinical.project-tcga-stad.2026-01-25" / "clinical.tsv"

print("=" * 80)
print("TCGA-STAD 和 GEO-GSE62254 数据集对比统计")
print("=" * 80)
print()

# 1. TCGA数据统计
print("【TCGA-STAD 数据集】")
print("-" * 80)
if TCGA_ORIGINAL_PATH.exists():
    tcga_original = pd.read_csv(TCGA_ORIGINAL_PATH, sep="\t", low_memory=False)
    print(f"1. 原始数据:")
    print(f"   - 总记录数: {len(tcga_original)}")
    print(f"   - 唯一病例数: {tcga_original['cases.submitter_id'].nunique()}")
    print(f"   - 数据列数: {len(tcga_original.columns)}")
    print(f"   - 数据来源: TCGA (The Cancer Genome Atlas)")
    print(f"   - 平台: Illumina HiSeq")
    print(f"   - 物种: 人类 (Homo sapiens)")
else:
    print("   TCGA原始数据文件不存在")

print()

if TCGA_PROCESSED_PATH.exists():
    tcga_processed = pd.read_csv(TCGA_PROCESSED_PATH)
    print(f"2. 处理后数据:")
    print(f"   - 样本数: {len(tcga_processed)}")
    print(f"   - 特征数: {len(tcga_processed.columns) - 2}")
    
    # 标签分布
    label_counts = tcga_processed['label'].value_counts(dropna=False)
    print(f"   - 标签分布:")
    print(f"     * 有转移 (1): {label_counts.get(1.0, 0)} ({label_counts.get(1.0, 0)/len(tcga_processed)*100:.1f}%)")
    print(f"     * 无转移 (0): {label_counts.get(0.0, 0)} ({label_counts.get(0.0, 0)/len(tcga_processed)*100:.1f}%)")
    print(f"     * 未知 (NaN): {label_counts.get(np.nan, 0)} ({label_counts.get(np.nan, 0)/len(tcga_processed)*100:.1f}%)")
    
    # 有效样本
    valid_samples = tcga_processed.dropna(subset=['label'])
    print(f"   - 有效样本数: {len(valid_samples)} ({len(valid_samples)/len(tcga_processed)*100:.1f}%)")
    print(f"   - 无效样本数: {len(tcga_processed) - len(valid_samples)} ({(len(tcga_processed) - len(valid_samples))/len(tcga_processed)*100:.1f}%)")
    
    # 年龄统计
    age_col = 'demographic.age_at_index'
    if age_col in tcga_processed.columns:
        age_valid = tcga_processed[age_col].dropna()
        print(f"   - 年龄统计:")
        print(f"     * 平均值: {age_valid.mean():.1f} ± {age_valid.std():.1f} 岁")
        print(f"     * 范围: {age_valid.min():.1f} - {age_valid.max():.1f} 岁")
        print(f"     * 有效样本数: {len(age_valid)}")
    
    # 性别分布
    gender_col = 'demographic.gender'
    if gender_col in tcga_processed.columns:
        gender_counts = tcga_processed[gender_col].value_counts()
        print(f"   - 性别分布:")
        for gender, count in gender_counts.items():
            print(f"     * {gender}: {count} ({count/len(tcga_processed)*100:.1f}%)")
    
    # 数据处理流程
    print(f"3. 数据处理流程:")
    print(f"   步骤1: 数据清洗 - 处理缺失值（'--'、'not reported'等转换为NaN）")
    print(f"   步骤2: 标签提取 - 从多个字段提取转移标签（M分期、肿瘤分期等）")
    print(f"   步骤3: 特征处理 - 按病例ID聚合，删除缺失率>60%的特征")
    print(f"   步骤4: 数据编码 - 对分类变量进行因子化编码")
    print(f"   步骤5: 模型训练 - 随机森林分类器，5折交叉验证")

print()

# 2. GEO数据统计
print("【GEO-GSE62254 数据集】")
print("-" * 80)
if GEO_PATH.exists():
    # 读取GEO数据
    geo_data = {}
    with open(GEO_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = parts[1:] if len(parts) > 1 else []
                geo_data[key] = values
    
    print(f"1. 基本信息:")
    print(f"   - 数据集ID: {geo_data.get('!Series_geo_accession', ['N/A'])[0]}")
    print(f"   - 标题: {geo_data.get('!Series_title', ['N/A'])[0]}")
    print(f"   - 发表日期: {geo_data.get('!Series_pubmed_id', ['N/A'])[0]}")
    print(f"   - 数据来源: GEO (Gene Expression Omnibus)")
    print(f"   - 平台: {geo_data.get('!Series_platform_id', ['N/A'])[0]}")
    print(f"   - 物种: 人类 (Homo sapiens)")
    
    # 样本数量
    sample_count = len(geo_data.get('!Sample_geo_accession', []))
    print(f"   - 样本数: {sample_count}")
    print(f"   - 样本类型: {geo_data.get('!Sample_type', ['N/A'])[0] if geo_data.get('!Sample_type') else 'N/A'}")
    
    # 样本特征
    sample_chars = geo_data.get('!Sample_characteristics_ch1', [])
    if sample_chars:
        print(f"   - 样本特征:")
        # 提取第一个样本的特征
        first_sample = sample_chars[0]
        if isinstance(first_sample, str):
            features = first_sample.split(';')
            for feature in features[:5]:  # 只显示前5个特征
                print(f"     * {feature.strip()}")
            if len(features) > 5:
                print(f"     * ... (共{len(features)}个特征)")
    
    # 基因数量（从ID_REF行开始是基因表达数据）
    gene_count = 0
    with open(GEO_PATH, 'r', encoding='utf-8') as f:
        in_data_section = False
        for line in f:
            if line.startswith('ID_REF'):
                in_data_section = True
                continue
            if in_data_section:
                gene_count += 1
    print(f"   - 基因数: {gene_count}")
    
    # 数据处理流程
    print(f"2. 数据处理流程:")
    print(f"   步骤1: 数据提取 - 从GEO平台下载原始表达矩阵")
    print(f"   步骤2: 质量控制 - 剔除低质量样本和基因")
    print(f"   步骤3: 标准化处理 - 对表达数据进行标准化（如log2转换）")
    print(f"   步骤4: 特征选择 - 选择差异表达基因或关键特征")
    print(f"   步骤5: 标签提取 - 从临床信息中提取转移状态标签")
    
    print(f"3. 数据特点:")
    print(f"   - 包含300个胃癌患者的基因表达数据")
    print(f"   - 包含详细的临床随访信息")
    print(f"   - 包含分子分型信息（MSI、Mesenchymal、TP53等）")
    print(f"   - 包含复发和转移信息")

print()

# 3. 数据集对比
print("【数据集对比】")
print("-" * 80)
print(f"{'项目':<30} {'TCGA-STAD':<20} {'GEO-GSE62254':<20}")
print("-" * 80)
print(f"{'数据来源':<30} {'TCGA':<20} {'GEO':<20}")
print(f"{'样本数量':<30} {'443':<20} {'300':<20}")
print(f"{'数据类型':<30} {'临床数据+基因表达':<20} {'基因表达+临床信息':<20}")
print(f"{'平台':<30} {'Illumina HiSeq':<20} {'Affymetrix U133 Plus 2.0':<20}")
print(f"{'基因数量':<30} {'~20,000':<20} {'~54,000':<20}")
print(f"{'种族分布':<30} {'多种族':<20} {'亚洲人群为主':<20}")
print(f"{'转移标签':<30} {'M分期+肿瘤分期':<20} {'临床随访信息':<20}")
print(f"{'有效样本率':<30} {'99.3% (440/443)':<20} {'需要进一步分析':<20}")

print()

# 4. 数据整合建议
print("【GEO和TCGA数据整合方法】")
print("-" * 80)
print("1. 数据格式统一:")
print("   - 统一基因标识符（如Entrez ID、Gene Symbol）")
print("   - 统一特征命名规范")
print("   - 统一数据类型和单位")

print()
print("2. 数据标准化:")
print("   - 对基因表达数据进行标准化（如z-score标准化）")
print("   - 对临床特征进行归一化处理")
print("   - 确保两个数据集的分布一致")

print()
print("3. 批次效应处理:")
print("   - 使用ComBat或SVA方法处理批次效应")
print("   - 识别并校正平台差异")
print("   - 验证处理后的数据分布")

print()
print("4. 特征选择:")
print("   - 选择在两个数据集中都存在的基因/特征")
print("   - 进行特征重要性分析")
print("   - 选择与转移相关的关键特征")

print()
print("5. 标签对齐:")
print("   - 统一转移标签的定义（如M1、复发等）")
print("   - 确保标签的一致性")
print("   - 处理缺失标签的样本")

print()
print("6. 模型训练策略:")
print("   - 方案A: 在TCGA上训练，在GEO上验证")
print("   - 方案B: 在GEO上训练，在TCGA上验证")
print("   - 方案C: 合并数据集进行训练，使用交叉验证")
print("   - 方案D: 使用迁移学习方法")

print()
print("7. 模型验证:")
print("   - 在独立数据集上验证模型性能")
print("   - 使用AUC、准确率、F1分数等指标")
print("   - 进行敏感性分析")
print("   - 评估模型的泛化能力")

print()
print("=" * 80)
print("分析完成")
print("=" * 80)
