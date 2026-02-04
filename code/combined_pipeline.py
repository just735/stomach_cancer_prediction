"""
乳腺癌远处转移预测模型 - 完整分析流程
整合了从数据整合到模型训练的7个步骤

步骤1：数据整合
步骤2：数据预处理
步骤3：差异基因筛选
步骤4：WGCNA分析
步骤5：靶基因筛选
步骤6：LASSO回归分析
步骤7：模型训练
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import tempfile
import warnings
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("警告: statsmodels未安装，将使用简化的多重检验校正方法")

# 使用Random Forest作为最佳模型

# Filter warnings
warnings.filterwarnings('ignore')

# Global Path Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR_BASE = BASE_DIR / "output"

def step1_data_integration():
    """步骤1：数据整合"""
    print("\n" + "=" * 60)
    print("步骤1：数据整合")
    print("=" * 60)
    
    OUTPUT_DIR = DATA_DIR / "integrated"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. GEO数据集列表（优先使用已处理的数据）
    geo_datasets = {
        "GSE15459": {
            "processed": DATA_DIR / "GEO" / "geo_processed.csv",
            "expr": DATA_DIR / "GEO" / "GSE15459_series_matrix.txt",
            "clin": DATA_DIR / "GEO" / "GSE15459_outcome.xls",
        },
    }
    
    # 2. TCGA-STAD数据
    tcga_stad = {
        "processed": DATA_DIR / "STAD" / "stad_processed.csv"
    }
    
    all_datasets = []
    
    # 3. 处理GEO数据集（优先使用已处理的数据）
    print("\n[1/2] 处理GEO数据集...")
    for dataset_name, paths in geo_datasets.items():
        print(f"\n处理 {dataset_name}...")
        
        # 优先使用已处理的数据
        if paths["processed"].exists():
            try:
                geo_data = pd.read_csv(paths["processed"], index_col=0)
                # 检查是否有label列
                if 'label' in geo_data.columns:
                    geo_data = geo_data.rename(columns={'label': 'metastasis'})
                elif 'metastasis_label' in geo_data.columns:
                    geo_data = geo_data.rename(columns={'metastasis_label': 'metastasis'})
                
                # 只保留有转移标签的样本
                if 'metastasis' in geo_data.columns:
                    geo_data = geo_data[geo_data['metastasis'].notna()].copy()
                    if len(geo_data) > 0:
                        geo_data['dataset'] = dataset_name
                        all_datasets.append(geo_data)
                        print(f"  ✓ 成功: 从已处理数据读取 {len(geo_data)} 个样本")
                    else:
                        print(f"  跳过: 已处理数据中无有效转移标签样本")
                else:
                    print(f"  跳过: 已处理数据中未找到转移标签列")
            except Exception as e:
                print(f"  错误: 读取已处理数据失败: {e}")
                print(f"  尝试从原始数据处理...")
                # 如果已处理数据读取失败，尝试从原始数据处理
                if paths["expr"].exists() and paths["clin"].exists():
                    try:
                        # 使用与process_geo_data.py相同的逻辑
                        expr = pd.read_csv(paths["expr"], sep="\t", comment="!", index_col=0, engine="python", on_bad_lines="skip")
                        expr_t = expr.T.reset_index().rename(columns={"index": "sample_id"})
                        
                        # 排除不合格样本
                        excluded = ["GSM387788", "GSM387790", "GSM387793", "GSM387797",
                                    "GSM387798", "GSM387799", "GSM387842"]
                        expr_t = expr_t[~expr_t["sample_id"].isin(excluded)].copy()
                        
                        # 读取临床数据
                        suf = paths["clin"].suffix.lower()
                        if suf == ".xls":
                            clinical = pd.read_excel(paths["clin"], engine="xlrd")
                        elif suf == ".xlsx":
                            clinical = pd.read_excel(paths["clin"], engine="openpyxl")
                        else:
                            clinical = pd.read_csv(paths["clin"], sep="\t")
                        
                        # 找到样本ID列
                        gsm_col = [c for c in clinical.columns if "GSM" in str(c) or "sample" in str(c).lower()]
                        if not gsm_col:
                            print(f"  跳过: 未找到样本ID列")
                            continue
                        gsm_col = gsm_col[0]
                        
                        # 使用完善的转移标签提取逻辑（与process_geo_data.py一致）
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
                            
                            # 方法2: 从Outcome字段推断
                            outcome_cols = [c for c in clinical.columns if "outcome" in str(c).lower()]
                            for col in outcome_cols:
                                if pd.notna(row.get(col)):
                                    val = str(row[col]).lower()
                                    if "1" in val or "dead" in val:
                                        return 1
                                    elif "0" in val or "alive" in val:
                                        return 0
                            
                            # 方法3: 从Stage字段推断
                            stage_cols = [c for c in clinical.columns if "stage" in str(c).lower()]
                            for col in stage_cols:
                                if pd.notna(row.get(col)):
                                    val = str(row[col]).upper()
                                    if "IV" in val or "4" in val:
                                        return 1
                            
                            return np.nan
                        
                        clinical['metastasis'] = clinical.apply(extract_metastasis_label, axis=1)
                        
                        # 合并数据
                        merged = pd.merge(expr_t, clinical[[gsm_col, 'metastasis']], 
                                         left_on="sample_id", right_on=gsm_col, how="inner")
                        
                        merged = merged[merged['metastasis'].notna()].copy()
                        
                        if len(merged) > 0:
                            merged['dataset'] = dataset_name
                            merged = merged.set_index("sample_id", drop=True)
                            all_datasets.append(merged)
                            print(f"  ✓ 成功: 从原始数据处理 {len(merged)} 个样本")
                        else:
                            print(f"  跳过: 无有效转移标签样本")
                    except Exception as e:
                        print(f"  错误: 处理原始数据时出错: {e}")
                        continue
        else:
            print(f"  跳过: 已处理数据文件不存在，且原始数据文件不完整")
            continue

    # 4. 处理TCGA-STAD数据（使用已处理的数据）
    print("\n[2/2] 处理TCGA-STAD数据...")
    if tcga_stad["processed"].exists():
        try:
            stad_data = pd.read_csv(tcga_stad["processed"], index_col=0)
            # 检查并重命名标签列
            if 'label' in stad_data.columns:
                stad_data = stad_data.rename(columns={'label': 'metastasis'})
            elif 'metastasis_label' in stad_data.columns:
                stad_data = stad_data.rename(columns={'metastasis_label': 'metastasis'})
            
            # 只保留有转移标签的样本
            if 'metastasis' in stad_data.columns:
                stad_data = stad_data[stad_data['metastasis'].notna()].copy()
                if len(stad_data) > 0:
                    stad_data['dataset'] = 'TCGA-STAD'
                    all_datasets.append(stad_data)
                    print(f"  ✓ 成功: {len(stad_data)} 个样本")
                else:
                    print(f"  跳过: 无有效转移标签样本")
            else:
                print(f"  跳过: 未找到转移标签列")
        except Exception as e:
            print(f"  错误: 处理TCGA-STAD时出错: {e}")
    else:
        print(f"  跳过: TCGA-STAD已处理数据文件不存在")

    # 5. 合并所有数据集
    if len(all_datasets) == 0:
        print("\n错误: 没有成功处理任何数据集")
        return False

    print(f"\n合并 {len(all_datasets)} 个数据集...")
    
    # 检查所有数据集是否有metastasis列
    for i, dataset in enumerate(all_datasets):
        if 'metastasis' not in dataset.columns:
            print(f"  警告: 数据集 {i+1} 缺少metastasis列")
    
    # 合并数据（使用outer join以保留所有特征）
    combined_data = pd.concat(all_datasets, axis=0, join='outer', sort=False)
    
    # 确保metastasis列存在
    if 'metastasis' not in combined_data.columns:
        print("错误: 合并后未找到metastasis列")
        return False
    
    # 只保留有转移标签的样本
    combined_data = combined_data[combined_data['metastasis'].notna()].copy()
    
    # 确保metastasis是数值型（0或1）
    combined_data['metastasis'] = pd.to_numeric(combined_data['metastasis'], errors='coerce')
    combined_data = combined_data[combined_data['metastasis'].notna()].copy()
    
    # 确保metastasis只包含0和1
    valid_labels = combined_data['metastasis'].isin([0, 1])
    if not valid_labels.all():
        print(f"  警告: 发现 {(~valid_labels).sum()} 个非0/1的标签值，将被排除")
        combined_data = combined_data[valid_labels].copy()
    
    print(f"合并后总样本数: {len(combined_data)}")
    print(f"合并后总特征数: {combined_data.shape[1]}")
    
    if 'metastasis' in combined_data.columns:
        label_counts = combined_data['metastasis'].value_counts()
        print(f"  无转移 (0): {label_counts.get(0, 0)}")
        print(f"  有转移 (1): {label_counts.get(1, 0)}")
        
        # 检查数据集来源分布
        if 'dataset' in combined_data.columns:
            dataset_counts = combined_data['dataset'].value_counts()
            print(f"\n数据集来源分布:")
            for dataset_name, count in dataset_counts.items():
                print(f"  {dataset_name}: {count} 个样本")
    
    # 验证数据有效性
    if len(combined_data) == 0:
        print("错误: 合并后没有有效样本")
        return False
    
    if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
        print("警告: 合并后只有一个类别的样本，可能影响模型训练")
    
    # 保存整合后的数据
    output_path = OUTPUT_DIR / "integrated_data.csv"
    combined_data.to_csv(output_path, index=True)
    print(f"\n整合数据已保存: {output_path}")
    print(f"数据形状: {combined_data.shape[0]} 个样本, {combined_data.shape[1]} 个特征")
    return True

def step2_data_preprocessing():
    """步骤2：数据预处理"""
    print("\n" + "=" * 60)
    print("步骤2：数据预处理")
    print("=" * 60)
    
    INPUT_FILE = DATA_DIR / "integrated" / "integrated_data.csv"
    OUTPUT_DIR = DATA_DIR / "preprocessed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] 读取整合数据...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"整合数据文件不存在: {INPUT_FILE}")
    
    data = pd.read_csv(INPUT_FILE, index_col=0)
    print(f"  读取数据: {data.shape[0]} 个样本, {data.shape[1]} 个特征")
    
    expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
    metadata_cols = [c for c in data.columns if c in ['metastasis', 'dataset']]
    
    X_expr = data[expr_cols].copy()
    metadata = data[metadata_cols].copy()
    
    print("\n[2/4] 处理基因标识和特征类型...")
    print("  提示: 假设数据已经是基因Symbol")
    
    # 分离数值型和分类型特征
    numeric_cols = []
    categorical_cols = []
    
    for col in X_expr.columns:
        try:
            # 尝试转换为数值型
            pd.to_numeric(X_expr[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # 无法转换为数值型，视为分类变量
            categorical_cols.append(col)
    
    print(f"  数值型特征: {len(numeric_cols)} 个")
    print(f"  分类型特征: {len(categorical_cols)} 个")
    
    X_numeric = X_expr[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_expr.index)
    X_categorical = X_expr[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=X_expr.index)
    
    print("\n[3/4] 异常值检测...")
    # 只对数值型特征进行异常值检测
    if len(numeric_cols) == 0:
        print("  警告: 没有数值型特征，跳过异常值检测")
        X_expr_clean = X_expr.copy()
        metadata_clean = metadata.copy()
    else:
        if X_numeric.shape[1] > 1000:
            cv = X_numeric.std() / (X_numeric.mean().abs() + 1e-10)
            top_genes = cv.nlargest(int(X_numeric.shape[1] * 0.25)).index
            X_cluster = X_numeric[top_genes].copy()
            print(f"  使用前25%变异基因进行聚类: {len(top_genes)} 个基因")
        else:
            X_cluster = X_numeric.copy()
            print(f"  使用全部数值型特征进行聚类: {X_cluster.shape[1]} 个特征")
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_cluster),
            index=X_cluster.index,
            columns=X_cluster.columns
        )
        
        # 检查并处理无效值
        if X_scaled.isna().any().any() or np.isinf(X_scaled.values).any():
            print("  警告: 发现NaN或Inf值，进行填充...")
            # 用0填充NaN和Inf
            X_scaled = X_scaled.fillna(0)
            X_scaled = X_scaled.replace([np.inf, -np.inf], 0)
        
        # 验证数据有效性
        if not np.isfinite(X_scaled.values).all():
            print("  警告: 标准化后仍有无效值，跳过异常值检测")
            X_expr_clean = X_expr.copy()
            metadata_clean = metadata.copy()
        else:
            print("  计算样本间距离矩阵...")
            try:
                distance_matrix = pdist(X_scaled.values, metric='euclidean')
                
                # 检查距离矩阵中的无效值
                if not np.isfinite(distance_matrix).all():
                    print("  警告: 距离矩阵包含无效值，跳过异常值检测")
                    X_expr_clean = X_expr.copy()
                    metadata_clean = metadata.copy()
                else:
                    print("  进行层次聚类...")
                    linkage_matrix = linkage(distance_matrix, method='ward')
                    
                    # 绘制聚类树
                    dendro_path = OUTPUT_DIR / "sample_dendrogram.png"
                    plt.figure(figsize=(15, 8))
                    dendrogram(linkage_matrix, labels=X_scaled.index, leaf_rotation=90, leaf_font_size=8)
                    plt.title("Sample Clustering Dendrogram (Before Outlier Removal)")
                    plt.xlabel("Sample ID")
                    plt.ylabel("Distance")
                    plt.tight_layout()
                    plt.savefig(dendro_path, dpi=150)
                    plt.close()
                    print(f"  聚类树已保存: {dendro_path}")
                    
                    # 异常值检测
                    cutHeight = 150
                    clusters = fcluster(linkage_matrix, t=cutHeight, criterion='distance')
                    cluster_counts = pd.Series(clusters).value_counts()
                    outlier_clusters = cluster_counts[cluster_counts <= 2].index
                    
                    outlier_indices = []
                    for cluster_id in outlier_clusters:
                        outlier_indices.extend(X_scaled.index[clusters == cluster_id].tolist())
                    
                    if len(outlier_indices) > 0:
                        print(f"  检测到 {len(outlier_indices)} 个异常样本")
                        X_expr_clean = X_expr.drop(index=outlier_indices)
                        metadata_clean = metadata.drop(index=outlier_indices)
                    else:
                        print("  未检测到异常样本")
                        X_expr_clean = X_expr.copy()
                        metadata_clean = metadata.copy()
            except Exception as e:
                print(f"  警告: 异常值检测过程出错: {e}")
                print("  跳过异常值检测，使用所有样本")
                X_expr_clean = X_expr.copy()
                metadata_clean = metadata.copy()
    
    print("\n[4/4] 数据标准化...")
    # 只对数值型特征进行标准化，分类特征需要编码
    if len(numeric_cols) > 0:
        X_numeric_clean = X_expr_clean[numeric_cols].copy()
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_numeric_clean = pd.DataFrame(
            imputer.fit_transform(X_numeric_clean),
            index=X_numeric_clean.index,
            columns=X_numeric_clean.columns
        )
        scaler_final = StandardScaler()
        X_numeric_normalized = pd.DataFrame(
            scaler_final.fit_transform(X_numeric_clean),
            index=X_numeric_clean.index,
            columns=X_numeric_clean.columns
        )
    else:
        X_numeric_normalized = pd.DataFrame(index=X_expr_clean.index)
    
    # 对分类特征进行编码（使用LabelEncoder）
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        X_categorical_clean = X_expr_clean[categorical_cols].copy()
        X_categorical_encoded = pd.DataFrame(index=X_categorical_clean.index)
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_categorical_encoded[col] = le.fit_transform(X_categorical_clean[col].astype(str))
    else:
        X_categorical_encoded = pd.DataFrame(index=X_expr_clean.index)
    
    # 合并数值型和编码后的分类特征
    X_normalized = pd.concat([X_numeric_normalized, X_categorical_encoded], axis=1)
    X_normalized = X_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
    processed_data = pd.concat([X_normalized, metadata_clean], axis=1)
    output_path = OUTPUT_DIR / "preprocessed_data.csv"
    processed_data.to_csv(output_path, index=True)
    print(f"\n预处理数据已保存: {output_path}")
    metadata_clean.to_csv(OUTPUT_DIR / "metadata.csv")
    return True

def step3_differential_genes():
    """步骤3：差异基因筛选"""
    print("\n" + "=" * 60)
    print("步骤3：差异基因筛选")
    print("=" * 60)
    
    INPUT_FILE = DATA_DIR / "preprocessed" / "preprocessed_data.csv"
    OUTPUT_DIR = DATA_DIR / "differential_genes"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] 读取预处理数据...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"预处理数据文件不存在: {INPUT_FILE}")
    
    data = pd.read_csv(INPUT_FILE, index_col=0)
    expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
    X = data[expr_cols].copy()
    y = data['metastasis'].copy()
    
    r_available = False
    try:
        result = subprocess.run(["R", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            r_available = True
            print("\n[2/3] 检测到R环境，将使用limma进行差异分析")
    except:
        print("\n[2/3] 未检测到R环境，将使用Python的t-test进行差异分析")
    
    if r_available:
        print("  使用limma包进行差异分析...")
        r_script = f"""
library(limma)
data <- read.csv("{INPUT_FILE.as_posix()}", row.names=1, check.names=FALSE)
expr_cols <- setdiff(colnames(data), c("metastasis", "dataset"))
expr_data <- as.matrix(data[, expr_cols])
group <- factor(data$metastasis, levels=c(0, 1), labels=c("non_metastasis", "metastasis"))
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)
fit <- lmFit(expr_data, design)
contrast.matrix <- makeContrasts(metastasis - non_metastasis, levels=design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)
results <- topTable(fit2, number=Inf, adjust.method="BH", p.value=0.05, lfc=0.5)
write.csv(results, "{(OUTPUT_DIR / 'limma_results.csv').as_posix()}", row.names=TRUE)
"""
        r_script_path = OUTPUT_DIR / "limma_analysis.R"
        with open(r_script_path, 'w', encoding='utf-8') as f:
            f.write(r_script)
        
        try:
            result = subprocess.run(["Rscript", str(r_script_path)], capture_output=True, text=True, timeout=300, cwd=str(OUTPUT_DIR))
            if result.returncode != 0:
                print(f"  R脚本执行失败: {result.stderr}")
                r_available = False
        except Exception as e:
            print(f"  运行R脚本时出错: {e}")
            r_available = False

    if not r_available:
        print("\n[2/3] 使用Python进行差异分析...")
        group0 = X[y == 0]
        group1 = X[y == 1]
        
        deg_results = []
        for gene in X.columns:
            try:
                stat, pval = ttest_ind(group0[gene], group1[gene], equal_var=False)
                mean0 = group0[gene].mean()
                mean1 = group1[gene].mean()
                logFC = np.log2((mean1 + 1e-10) / (mean0 + 1e-10)) if mean0 > 0 else 0
                
                deg_results.append({
                    'gene': gene,
                    'logFC': logFC,
                    'pvalue': pval
                })
            except:
                continue
        
        deg_df = pd.DataFrame(deg_results)
        
        # 多重检验校正
        if STATSMODELS_AVAILABLE:
            _, deg_df['adj_pvalue'], _, _ = multipletests(deg_df['pvalue'], method='fdr_bh')
        else:
            # 简化的FDR校正（Bonferroni方法）
            n_tests = len(deg_df)
            deg_df['adj_pvalue'] = deg_df['pvalue'] * n_tests
            deg_df['adj_pvalue'] = deg_df['adj_pvalue'].clip(upper=1.0)
        
        deg_df = deg_df[(np.abs(deg_df['logFC']) > 0.5) & (deg_df['adj_pvalue'] < 0.05)].copy()
        
        deg_df.to_csv(OUTPUT_DIR / "differential_genes.csv", index=False)
        deg_genes = deg_df['gene'].tolist()
        deg_up = deg_df[deg_df['logFC'] > 0.5]['gene'].tolist()
        deg_down = deg_df[deg_df['logFC'] < -0.5]['gene'].tolist()
    else:
        if (OUTPUT_DIR / "limma_results.csv").exists():
            limma_results = pd.read_csv(OUTPUT_DIR / "limma_results.csv", index_col=0)
            deg_genes = limma_results.index.tolist()
            deg_up = limma_results[limma_results['logFC'] > 0.5].index.tolist()
            deg_down = limma_results[limma_results['logFC'] < -0.5].index.tolist()
        else:
            deg_genes = []
            deg_up = []
            deg_down = []

    print(f"  上调基因数: {len(deg_up)}")
    print(f"  下调基因数: {len(deg_down)}")
    
    with open(OUTPUT_DIR / "deg_genes_list.txt", 'w') as f:
        f.write('\n'.join(deg_genes))
    with open(OUTPUT_DIR / "deg_up_genes.txt", 'w') as f:
        f.write('\n'.join(deg_up))
    with open(OUTPUT_DIR / "deg_down_genes.txt", 'w') as f:
        f.write('\n'.join(deg_down))
    
    print(f"\n差异基因列表已保存")
    return True

def step4_wgcna_analysis():
    """步骤4：WGCNA分析"""
    print("\n" + "=" * 60)
    print("步骤4：WGCNA分析")
    print("=" * 60)
    
    INPUT_FILE = DATA_DIR / "preprocessed" / "preprocessed_data.csv"
    OUTPUT_DIR = DATA_DIR / "wgcna"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/5] 读取数据...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"预处理数据文件不存在: {INPUT_FILE}")
    
    data = pd.read_csv(INPUT_FILE, index_col=0)
    expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
    X = data[expr_cols].copy()
    y = data['metastasis'].copy()
    
    print("\n[2/5] 选择高变异基因...")
    cv = X.std() / (X.mean().abs() + 1e-10)
    top_genes = cv.nlargest(int(X.shape[1] * 0.25)).index
    X_selected = X[top_genes].copy()
    print(f"  选择前25%变异基因: {len(top_genes)} 个基因")
    
    print("\n[3/5] 确定软阈值...")
    def calculate_soft_threshold(X, powers=np.arange(1, 21)):
        results = []
        for power in powers:
            adj = np.power(np.abs(np.corrcoef(X.T)), power)
            np.fill_diagonal(adj, 0)
            connectivity = adj.sum(axis=1)
            
            if len(connectivity) > 0 and connectivity.std() > 0:
                log_conn = np.log(connectivity + 1)
                mean_conn = log_conn.mean()
                ss_res = ((log_conn - mean_conn) ** 2).sum()
                ss_tot = ((log_conn - mean_conn) ** 2).sum()
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            else:
                r_squared = 0
            
            results.append({
                'power': power,
                'r_squared': r_squared,
                'mean_connectivity': connectivity.mean()
            })
        return pd.DataFrame(results)

    if len(X_selected) > 500:
        X_threshold = X_selected.iloc[:, :500]
    else:
        X_threshold = X_selected
        
    threshold_results = calculate_soft_threshold(X_threshold.values.T)
    
    optimal_power = 6
    valid_powers = threshold_results[threshold_results['r_squared'] > 0.9]
    if len(valid_powers) > 0:
        valid_powers = valid_powers.sort_values('mean_connectivity')
        optimal_power = int(valid_powers.iloc[0]['power'])
        print(f"  最优软阈值: β = {optimal_power}")
    else:
        print(f"  使用默认软阈值: β = {optimal_power}")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(threshold_results['power'], threshold_results['r_squared'], 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.axvline(x=optimal_power, color='g', linestyle='--')
    plt.title('Scale Free Topology Model Fit (R²)')
    plt.subplot(1, 2, 2)
    plt.plot(threshold_results['power'], threshold_results['mean_connectivity'], 'o-')
    plt.axvline(x=optimal_power, color='g', linestyle='--')
    plt.title('Mean Connectivity')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "soft_threshold_selection.png", dpi=150)
    plt.close()
    
    print("\n[4/5] 构建共表达网络和模块...")
    # 检查数据有效性
    if X_selected.isna().any().any() or np.isinf(X_selected.values).any():
        print("  警告: 发现NaN或Inf值，进行填充...")
        X_selected = X_selected.fillna(0)
        X_selected = X_selected.replace([np.inf, -np.inf], 0)
    
    # 计算邻接矩阵
    try:
        corr_matrix = np.corrcoef(X_selected.values.T)
        # 检查相关性矩阵中的无效值
        if not np.isfinite(corr_matrix).all():
            print("  警告: 相关性矩阵包含无效值，进行填充...")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        adj_matrix = np.power(np.abs(corr_matrix), optimal_power)
        np.fill_diagonal(adj_matrix, 0)
        
        # 转换为距离矩阵
        distance_matrix = 1 - adj_matrix
        # 确保距离矩阵值在合理范围内
        distance_matrix = np.clip(distance_matrix, 0, 2)
        
        # 检查距离矩阵有效性
        if not np.isfinite(distance_matrix).all():
            print("  警告: 距离矩阵包含无效值，进行填充...")
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=2.0, neginf=0.0)
        
        # 保存完整距离矩阵用于后续使用
        distance_matrix_full = distance_matrix.copy()
        
        distance_matrix = squareform(distance_matrix, checks=False)
        
        # 再次验证距离矩阵
        if not np.isfinite(distance_matrix).all():
            print("  警告: 压缩后的距离矩阵包含无效值，进行填充...")
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=2.0, neginf=0.0)
        
        linkage_matrix = linkage(distance_matrix, method='average')
    except Exception as e:
        print(f"  警告: 构建共表达网络时出错: {e}")
        print("  使用简化的模块划分方法...")
        # 如果构建网络失败，使用简单的聚类方法
        from sklearn.cluster import KMeans
        n_modules = min(10, max(2, X_selected.shape[1] // 30))
        kmeans = KMeans(n_clusters=n_modules, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_selected.values.T)
        modules = pd.DataFrame({
            'gene': X_selected.columns,
            'module': clusters
        })
        
        # 计算模块与转移的相关性
        print("\n[5/5] 计算模块-临床特征相关性...")
        module_eigengenes = []
        module_names = []
        
        for module_id in modules['module'].unique():
            module_genes = modules[modules['module'] == module_id]['gene'].tolist()
            if len(module_genes) > 0:
                module_expr = X_selected[module_genes]
                pca = PCA(n_components=1)
                eigengene = pca.fit_transform(module_expr.values)
                module_eigengenes.append(eigengene.flatten())
                module_names.append(f"Module_{module_id}")
        
        if len(module_eigengenes) > 0:
            eigengene_df = pd.DataFrame(
                np.array(module_eigengenes).T,
                index=X_selected.index,
                columns=module_names
            )
            
            correlations = []
            for module_name in module_names:
                corr, pval = pearsonr(eigengene_df[module_name], y)
                correlations.append({'module': module_name, 'correlation': corr, 'pvalue': pval})
            
            module_corr = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
            top_modules = module_corr.head(3)
            print(f"\n  选择核心模块: {', '.join(top_modules['module'].tolist())}")
            
            core_genes = []
            for module_name in top_modules['module']:
                module_id = int(module_name.split('_')[1])
                module_genes = modules[modules['module'] == module_id]['gene'].tolist()
                core_genes.extend(module_genes)
            
            with open(OUTPUT_DIR / "wgcna_core_modules_genes.txt", 'w') as f:
                f.write('\n'.join(core_genes))
            print(f"  核心模块基因已保存")
            
            # 热图
            plt.figure(figsize=(10, 6))
            corr_matrix = module_corr.set_index('module')[['correlation']].T
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
            plt.title('Module-Metastasis Correlation')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "module_metastasis_correlation.png", dpi=150)
            plt.close()
        
        return True
    
    # 如果成功构建了linkage_matrix，继续执行动态树切割
    # 动态树切割
    min_module_size = 30
    clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
    
    # 合并小模块
    cluster_counts = pd.Series(clusters).value_counts()
    small_clusters = cluster_counts[cluster_counts < min_module_size].index
    if len(small_clusters) > 0:
        for small_cluster in small_clusters:
            clusters[clusters == small_cluster] = -1
            
    # 重新分配未分类基因
    unassigned = np.where(clusters == -1)[0]
    if len(unassigned) > 0:
        # 将压缩的距离矩阵转换回完整矩阵
        from scipy.spatial.distance import squareform
        dist_matrix_full = squareform(distance_matrix)
        
        for idx in unassigned:
            if idx < len(dist_matrix_full):
                distances = dist_matrix_full[idx]
                assigned = np.where(clusters != -1)[0]
                if len(assigned) > 0:
                    nearest = assigned[np.argmin(distances[assigned])]
                    clusters[idx] = clusters[nearest]
    
    modules = pd.DataFrame({
        'gene': X_selected.columns,
        'module': clusters
    })
    
    print("\n[5/5] 计算模块-临床特征相关性...")
    module_eigengenes = []
    module_names = []
    
    for module_id in modules['module'].unique():
        module_genes = modules[modules['module'] == module_id]['gene'].tolist()
        if len(module_genes) > 0:
            module_expr = X_selected[module_genes]
            pca = PCA(n_components=1)
            eigengene = pca.fit_transform(module_expr.values)
            module_eigengenes.append(eigengene.flatten())
            module_names.append(f"Module_{module_id}")
    
    if len(module_eigengenes) > 0:
        eigengene_df = pd.DataFrame(
            np.array(module_eigengenes).T,
            index=X_selected.index,
            columns=module_names
        )
        
        correlations = []
        for module_name in module_names:
            corr, pval = pearsonr(eigengene_df[module_name], y)
            correlations.append({'module': module_name, 'correlation': corr, 'pvalue': pval})
        
        module_corr = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        top_modules = module_corr.head(3)
        print(f"\n  选择核心模块: {', '.join(top_modules['module'].tolist())}")
        
        core_genes = []
        for module_name in top_modules['module']:
            module_id = int(module_name.split('_')[1])
            module_genes = modules[modules['module'] == module_id]['gene'].tolist()
            core_genes.extend(module_genes)
        
        with open(OUTPUT_DIR / "wgcna_core_modules_genes.txt", 'w') as f:
            f.write('\n'.join(core_genes))
        print(f"  核心模块基因已保存")
        
        # 热图
        plt.figure(figsize=(10, 6))
        corr_matrix = module_corr.set_index('module')[['correlation']].T
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Module-Metastasis Correlation')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "module_metastasis_correlation.png", dpi=150)
        plt.close()
    
    return True

def step5_target_gene_selection():
    """步骤5：靶基因筛选"""
    print("\n" + "=" * 60)
    print("步骤5：靶基因筛选")
    print("=" * 60)
    
    DEG_DIR = DATA_DIR / "differential_genes"
    WGCNA_DIR = DATA_DIR / "wgcna"
    OUTPUT_DIR = DATA_DIR / "target_genes"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] 读取差异基因...")
    deg_genes_path = DEG_DIR / "deg_genes_list.txt"
    if not deg_genes_path.exists():
        raise FileNotFoundError(f"差异基因列表不存在")
    with open(deg_genes_path, 'r') as f:
        deg_genes = set([line.strip() for line in f if line.strip()])
    
    print("\n[2/3] 读取WGCNA核心模块基因...")
    wgcna_genes_path = WGCNA_DIR / "wgcna_core_modules_genes.txt"
    if not wgcna_genes_path.exists():
        raise FileNotFoundError(f"WGCNA核心模块基因列表不存在")
    with open(wgcna_genes_path, 'r') as f:
        wgcna_genes = set([line.strip() for line in f if line.strip()])
    
    print("\n[3/3] 计算交集...")
    target_genes = deg_genes & wgcna_genes
    print(f"  交集（靶基因）: {len(target_genes)}")
    
    if len(target_genes) == 0:
        print("  警告: 交集为空，将使用差异基因作为靶基因")
        target_genes = deg_genes
    
    with open(OUTPUT_DIR / "target_genes.txt", 'w') as f:
        f.write('\n'.join(sorted(target_genes)))
    print(f"\n靶基因列表已保存")
    return True

def step6_lasso_selection():
    """步骤6：LASSO回归分析"""
    print("\n" + "=" * 60)
    print("步骤6：LASSO回归分析")
    print("=" * 60)
    
    INPUT_FILE = DATA_DIR / "preprocessed" / "preprocessed_data.csv"
    TARGET_GENES_FILE = DATA_DIR / "target_genes" / "target_genes.txt"
    OUTPUT_DIR = DATA_DIR / "biomarkers"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/5] 读取数据...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"预处理数据文件不存在")
    data = pd.read_csv(INPUT_FILE, index_col=0)
    
    if TARGET_GENES_FILE.exists():
        with open(TARGET_GENES_FILE, 'r') as f:
            target_genes = set([line.strip() for line in f if line.strip()])
        available_genes = [g for g in target_genes if g in data.columns]
        if len(available_genes) > 0:
            X = data[available_genes].copy()
        else:
            expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
            X = data[expr_cols].copy()
    else:
        expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
        X = data[expr_cols].copy()
    
    y = data['metastasis'].copy()
    
    # 检查并处理缺失值
    print("\n[2/5] 处理缺失值...")
    if X.isna().any().any():
        print(f"  发现缺失值: {X.isna().sum().sum()} 个")
        # 使用中位数填充数值型特征
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        print(f"  已使用中位数填充缺失值")
    else:
        X_imputed = X.copy()
    
    # 检查y中是否有缺失值
    if y.isna().any():
        print(f"  警告: 标签中有缺失值，将排除这些样本")
        valid_mask = y.notna()
        X_imputed = X_imputed[valid_mask].copy()
        y = y[valid_mask].copy()
    
    # 验证数据有效性
    if X_imputed.isna().any().any() or np.isinf(X_imputed.values).any():
        print("  警告: 仍有无效值，进行填充...")
        X_imputed = X_imputed.fillna(0)
        X_imputed = X_imputed.replace([np.inf, -np.inf], 0)
    
    print("\n[3/5] 数据标准化...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        index=X_imputed.index,
        columns=X_imputed.columns
    )
    
    # 最终验证
    if X_scaled.isna().any().any() or np.isinf(X_scaled.values).any():
        print("  警告: 标准化后仍有无效值，进行填充...")
        X_scaled = X_scaled.fillna(0)
        X_scaled = X_scaled.replace([np.inf, -np.inf], 0)
    
    print("\n[4/5] LASSO回归分析...")
    alphas = np.logspace(-4, 1, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=10, max_iter=2000, random_state=42, n_jobs=-1)
    lasso_cv.fit(X_scaled.values, y.values)
    
    optimal_alpha = lasso_cv.alpha_
    print(f"  最优alpha: {optimal_alpha:.6f}")
    
    selected_features = X_scaled.columns[lasso_cv.coef_ != 0].tolist()
    print(f"  选中的生物标志物数: {len(selected_features)}")
    
    if len(selected_features) == 0:
        print("  警告: LASSO未选中任何特征，使用所有特征")
        selected_features = X_scaled.columns.tolist()
    
    print("\n[5/5] 绘制LASSO路径...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(lasso_cv.alphas_), lasso_cv.mse_path_.mean(axis=1), 'k-')
    plt.axvline(x=np.log10(optimal_alpha), color='r', linestyle='--')
    plt.title('LASSO Cross-Validation')
    
    plt.subplot(1, 2, 2)
    coef_paths = []
    for alpha in lasso_cv.alphas_[:20]:
        lasso = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        lasso.fit(X_scaled.values, y.values)
        coef_paths.append(lasso.coef_)
    
    plt.plot(np.log10(lasso_cv.alphas_[:20]), coef_paths)
    plt.axvline(x=np.log10(optimal_alpha), color='r', linestyle='--')
    plt.title('LASSO Coefficient Path')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lasso_path.png", dpi=150)
    plt.close()
    
    with open(OUTPUT_DIR / "biomarkers.txt", 'w') as f:
        f.write('\n'.join(selected_features))
    print(f"  生物标志物列表已保存")
    
    return True

def step7_model_training():
    """步骤7：模型训练"""
    print("\n" + "=" * 60)
    print("步骤7：模型训练")
    print("=" * 60)
    
    INPUT_FILE = DATA_DIR / "preprocessed" / "preprocessed_data.csv"
    BIOMARKERS_FILE = DATA_DIR / "biomarkers" / "biomarkers.txt"
    OUTPUT_DIR = BASE_DIR / "output" / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/6] 读取数据...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"预处理数据文件不存在")
    data = pd.read_csv(INPUT_FILE, index_col=0)
    
    if BIOMARKERS_FILE.exists():
        with open(BIOMARKERS_FILE, 'r') as f:
            biomarkers = [line.strip() for line in f if line.strip()]
        available_biomarkers = [b for b in biomarkers if b in data.columns]
        if len(available_biomarkers) > 0:
            X = data[available_biomarkers].copy()
        else:
            expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
            X = data[expr_cols].copy()
    else:
        expr_cols = [c for c in data.columns if c not in ['metastasis', 'dataset']]
        X = data[expr_cols].copy()
    
    y = data['metastasis'].copy()
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("\n[2/6] 数据拆分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n[3/6] 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    print("\n[4/6] 超参数优化...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 使用Random Forest（效果最好的模型）
    print("  优化随机森林模型...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train_scaled, y_train)
    best_model = rf_grid.best_estimator_
    print(f"    最优参数: {rf_grid.best_params_}")
    
    print("\n[5/6] 模型评估...")
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算详细指标
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    precision = report.get('1', {}).get('precision', 0)
    recall = report.get('1', {}).get('recall', 0)
    f1_score = report.get('1', {}).get('f1-score', 0)
    
    print(f"  Random Forest:")
    print(f"    准确率: {accuracy:.4f}")
    print(f"    ROC AUC: {roc_auc:.4f}")
    print(f"    精确率: {precision:.4f}")
    print(f"    召回率: {recall:.4f}")
    print(f"    F1分数: {f1_score:.4f}")
    
    print("\n[6/6] 保存结果...")
    # 保存模型评估结果
    results_df = pd.DataFrame({
        'Model': ['Random Forest'],
        'Accuracy': [accuracy],
        'ROC_AUC': [roc_auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1_score]
    })
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"  模型评估结果已保存: {OUTPUT_DIR / 'model_comparison.csv'}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print(f"  ROC曲线已保存: {OUTPUT_DIR / 'roc_curve.png'}")
    
    # 保存预测结果
    predictions = pd.DataFrame({
        'sample_id': X_test.index,
        'true_label': y_test.values,
        'predicted_label': y_pred,
        'predicted_proba': y_pred_proba
    })
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"  预测结果已保存: {OUTPUT_DIR / 'predictions.csv'}")
    
    print(f"\n最佳模型: Random Forest (ROC AUC = {roc_auc:.4f})")
    
    return True

def main():
    print("开始执行乳腺癌远处转移预测模型完整分析流程...")
    print(f"工作目录: {BASE_DIR}")
    
    try:
        if not step1_data_integration():
            print("步骤1失败，停止执行")
            return
        
        if not step2_data_preprocessing():
            print("步骤2失败，停止执行")
            return
            
        if not step3_differential_genes():
            print("步骤3失败，停止执行")
            return
            
        if not step4_wgcna_analysis():
            print("步骤4失败，停止执行")
            return
            
        if not step5_target_gene_selection():
            print("步骤5失败，停止执行")
            return
            
        if not step6_lasso_selection():
            print("步骤6失败，停止执行")
            return
            
        if not step7_model_training():
            print("步骤7失败，停止执行")
            return
            
        print("\n" + "=" * 60)
        print("所有步骤执行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
