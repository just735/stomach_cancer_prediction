# 胃癌远处转移预测研究

本仓库用于基于多数据集的胃癌远处转移预测研究。包含完整的数据整合、预处理、特征筛选和模型训练流程，旨在提供全面的胃癌预后预测和生物标志物发现平台。

## 项目概述

本研究旨在利用基因表达数据和临床数据构建胃癌远处转移预测模型，通过多种机器学习方法和生物信息学分析技术，探索潜在的预后生物标志物，为胃癌患者的个体化治疗和预后评估提供科学依据。

### 主要研究内容：

1. **数据整合**：整合多个GEO数据集和TCGA-STAD数据，提取转移相关样本
2. **数据预处理**：探针ID转换、异常值检测、数据标准化
3. **差异基因筛选**：使用limma或t-test识别差异表达基因
4. **WGCNA分析**：加权基因共表达网络分析，识别与转移相关的基因模块
5. **靶基因筛选**：取差异基因与WGCNA核心模块基因的交集
6. **LASSO回归**：进一步筛选生物标志物，剔除冗余特征
7. **模型训练**：使用多种机器学习算法（LR、RF、SVM、GBDT、XGBoost）进行训练和评估

## 完整分析流程

### 整合分析流程 (`code/combined_pipeline.py`)

完整的7步分析流程已整合到一个脚本中，按顺序执行所有步骤：

```powershell
python code/combined_pipeline.py
```

**流程包含的7个步骤**：

1. **数据整合**：整合多个GEO数据集和TCGA-STAD数据，提取转移相关样本
   - 读取GEO数据集的基因表达数据和临床数据
   - 读取TCGA-STAD的预处理数据
   - 提取转移状态标签
   - 合并所有数据集
   - 输出：`data/integrated/integrated_data.csv`

2. **数据预处理**：探针ID转换、异常值检测、数据标准化
   - 基于层次聚类的异常值检测
   - Z-score标准化
   - 输出：`data/preprocessed/preprocessed_data.csv`, `data/preprocessed/metadata.csv`, `data/preprocessed/sample_dendrogram.png`

3. **差异基因筛选**：使用limma或t-test识别差异表达基因
   - 优先使用R的limma包（如果R环境可用）
   - 否则使用Python的t-test
   - 筛选标准：|logFC| > 0.5 且 adj_pvalue < 0.05
   - 输出：`data/differential_genes/deg_genes_list.txt` 等

4. **WGCNA分析**：加权基因共表达网络分析
   - 选择变异系数前25%的基因
   - 确定最优软阈值（β）
   - 构建共表达网络和模块
   - 计算模块与转移的相关性
   - 筛选核心模块（相关性最高的2-3个）
   - 输出：`data/wgcna/wgcna_core_modules_genes.txt` 等

5. **靶基因筛选**：取差异基因与WGCNA核心模块基因的交集
   - 输出：`data/target_genes/target_genes.txt`

6. **LASSO回归**：进一步筛选生物标志物
   - 10折交叉验证选择最优λ
   - 筛选系数非零的基因作为生物标志物
   - 输出：`data/biomarkers/biomarkers.txt` 等

7. **模型训练**：使用多种机器学习模型进行训练和评估
   - 数据拆分（8:2训练/测试集）
   - 5折交叉验证进行超参数优化
   - 训练5种模型：逻辑回归(LR)、随机森林(RF)、支持向量机(SVM)、梯度提升决策树(GBDT)、XGBoost
   - 模型性能评估和比较
   - 输出：`output/models/model_comparison.csv`, `output/models/roc_curves.png` 等

**完整输出文件**：
- `data/integrated/integrated_data.csv` - 整合后的数据
- `data/preprocessed/preprocessed_data.csv` - 预处理后的数据
- `data/differential_genes/deg_genes_list.txt` - 差异基因列表
- `data/wgcna/wgcna_core_modules_genes.txt` - WGCNA核心模块基因
- `data/target_genes/target_genes.txt` - 靶基因列表
- `data/biomarkers/biomarkers.txt` - 最终生物标志物列表
- `output/models/model_comparison.csv` - 模型性能比较
- `output/models/best_model_predictions.csv` - 最佳模型预测结果

## 快速使用（简化流程）

如果只需要使用现有的最佳模型进行预测，可以使用以下简化流程：

### GEO数据集（胃癌基因表达数据）

**数据清洗**：`code/process_geo_data.py`
- 处理GEO数据库的基因表达数据和临床数据
- 输入：`data/GEO/GSE15459_series_matrix.txt` 和 `data/GEO/GSE15459_outcome.xls`
- 输出：`data/GEO/geo_processed.csv

**预测模型**：`code/predict_geo_metastasis.py`
- 使用随机森林分类器
- 自动特征选择（选择前1000个重要特征）
- 处理类别不平衡
- 输出：`output/GEO/predictions.csv`, `output/GEO/roc_curve.png`

**使用步骤**：
```powershell
# 1. 数据清洗
python code/process_geo_data.py

# 2. 运行预测
python code/predict_geo_metastasis.py
```

### TCGA-STAD数据集（胃癌临床数据）

**数据清洗**：`code/process_stad_data.py`
- 处理TCGA-STAD的临床和生物样本数据
- 输入：`data/STAD/clinical.project-tcga-stad.2026-01-25/clinical.tsv`
- 输出：`data/STAD/stad_processed.csv`

**预测模型**：`code/predict_stad_metastasis.py`
- 使用随机森林分类器
- 自动处理分类变量和缺失值
- 处理类别不平衡
- 输出：`output/STAD/predictions.csv`, `output/STAD/roc_curve.png`

**使用步骤**：
```powershell
# 1. 数据清洗
python code/process_stad_data.py

# 2. 运行预测
python code/predict_stad_metastasis.py
```

## GEO数据集详细说明

### 数据要求

确保以下文件存在于 `data/GEO/` 目录：
- `GSE15459_series_matrix.txt` - 基因表达数据
- `GSE15459_outcome.xls` 或 `GSE15459_outcome.xlsx` - 临床数据

### 数据格式

**基因表达数据** (`GSE15459_series_matrix.txt`):
- 格式：TSV文件
- 行：基因探针ID
- 列：样本ID（GSM编号）
- 值：基因表达值

**临床数据** (`GSE15459_outcome.xls`):
- 格式：Excel文件
- 必须包含：样本ID列（包含"GSM"）
- 可选列：Age_at_surgery, Gender, Laurenclassification, Stage, Outcome等

### 转移标签提取逻辑

脚本会从以下字段提取转移状态：
1. **直接转移字段**：包含"metastasis"的列
2. **Outcome字段**：如果Outcome=1或"dead"，可能暗示转移
3. **Stage字段**：Stage IV通常表示转移

### 预测模型特点

1. **自动特征选择**：
   - 如果特征数>10，自动选择前1000个最重要的特征
   - 使用f_classif进行特征选择

2. **处理类别不平衡**：
   - 自动使用class_weight='balanced'
   - 在参数搜索中尝试balanced和None

3. **参数优化**：
   - 使用随机搜索（RandomizedSearchCV）
   - 优化：criterion, n_estimators, max_depth, max_features, class_weight

## TCGA-STAD数据集详细说明

### 数据要求

**必需数据**：
- `data/STAD/clinical.project-tcga-stad.2026-01-25/clinical.tsv` - 临床数据
- `data/STAD/biospecimen.project-tcga-stad.2026-01-25/sample.tsv` - 生物样本数据

**可选数据**：
- **基因表达数据**：TCGA-STAD的RNA-seq或microarray数据
  - 当前脚本仅使用临床特征进行预测
  - 如需更高准确率，建议下载并整合基因表达数据

### 数据格式说明

**清洗后的数据格式**：
- **行**：样本（病例ID）
- **列**：特征 + 标签
  - 特征：年龄、性别、TNM分期、肿瘤分级等临床特征
  - 标签：`label` 列（0=无转移, 1=有转移）

**预测结果格式**：
- `true` - 真实标签
- `pred` - 预测标签
- `prob_positive` - 预测为转移的概率

### 注意事项

1. **数据限制**：
   - 当前数据集**仅包含临床特征**，不包含基因表达数据
   - 仅使用临床特征的预测准确率可能较低
   - 建议从TCGA下载RNA-seq数据并整合到数据集中

2. **类别不平衡**：
   - TCGA-STAD数据集中转移病例可能较少
   - 脚本会自动使用 `class_weight="balanced"` 处理类别不平衡
   - 使用分层抽样（stratify）保持训练/测试集的类别比例

3. **特征选择**：
   - 如果特征数 > 10，会自动进行特征选择（SelectKBest）
   - 最多选择100个最重要的特征

## 环境要求

### Python版本
- **推荐**：Python 3.10
- 不推荐使用Python 3.14+（可能存在兼容性问题）

### 必需依赖包

```powershell
pip install numpy==1.26.4
pip install pandas==2.0.3
pip install scikit-learn
pip install matplotlib==3.8.2
pip install pillow==10.0.0
pip install kiwisolver==1.4.5
pip install scipy
pip install statsmodels
```

### 可选依赖（用于读取Excel文件）

```powershell
pip install xlrd      # 用于读取.xls文件
pip install openpyxl  # 用于读取.xlsx文件
```

### 可选依赖（用于完整分析流程）

```powershell
pip install xgboost   # 用于XGBoost模型
```

### R环境（可选，用于limma差异分析）

如果使用R的limma包进行差异分析，需要：
1. 安装R（4.1.0及以上版本）
2. 安装R包：limma, BiocManager

## 目录结构

```
.
├── code/                          # 代码目录
│   ├── 1_data_integration.py      # 数据整合
│   ├── 2_data_preprocessing.py    # 数据预处理
│   ├── 3_differential_genes.py    # 差异基因筛选
│   ├── 4_wgcna_analysis.py        # WGCNA分析
│   ├── 5_target_gene_selection.py # 靶基因筛选
│   ├── 6_lasso_selection.py       # LASSO回归
│   ├── 7_model_training.py        # 模型训练
│   ├── process_geo_data.py        # GEO数据清洗（简化流程）
│   ├── predict_geo_metastasis.py  # GEO预测（简化流程）
│   ├── process_stad_data.py         # STAD数据清洗（简化流程）
│   └── predict_stad_metastasis.py # STAD预测（简化流程）
├── data/                          # 数据目录
│   ├── GEO/                       # GEO数据集
│   ├── STAD/                      # TCGA-STAD数据集
│   ├── integrated/                # 整合后的数据
│   ├── preprocessed/              # 预处理后的数据
│   ├── differential_genes/         # 差异基因结果
│   ├── wgcna/                     # WGCNA分析结果
│   ├── target_genes/              # 靶基因结果
│   └── biomarkers/                # 生物标志物结果
├── output/                        # 输出目录
│   ├── GEO/                       # GEO预测结果
│   ├── STAD/                      # STAD预测结果
│   └── models/                    # 模型训练结果
└── README.md                      # 本文件
```

## 输出文件说明

### 完整流程输出

- `data/integrated/integrated_data.csv` - 整合后的数据
- `data/preprocessed/preprocessed_data.csv` - 预处理后的数据
- `data/differential_genes/deg_genes_list.txt` - 差异基因列表
- `data/wgcna/wgcna_core_modules_genes.txt` - WGCNA核心模块基因
- `data/target_genes/target_genes.txt` - 靶基因列表
- `data/biomarkers/biomarkers.txt` - 最终生物标志物列表
- `output/models/model_comparison.csv` - 模型性能比较
- `output/models/best_model_predictions.csv` - 最佳模型预测结果

### 简化流程输出

每个数据集都会生成：
- `predictions.csv` - 测试集预测结果（包含真实标签、预测标签、预测概率）
- `predictions_all.csv` - 全部样本预测结果
- `roc_curve.png` - ROC曲线图（如果成功生成）

## 常见问题

### Q1: 数据清洗时找不到转移标签？

**原因**：临床数据中可能没有明确的转移字段

**解决**：
1. 检查临床数据文件，确认包含转移相关信息
2. 修改相应的数据清洗脚本中的转移标签提取函数
3. 或手动添加转移标签列到临床数据

### Q2: 预测时内存不足？

**原因**：基因表达数据特征数过多

**解决**：
1. 减少特征选择数量（修改 `k_select` 参数）
2. 减少随机搜索次数（修改 `n_iter` 参数）
3. 减少交叉验证折数（修改 `cv` 参数）

### Q3: ROC曲线生成失败？

**原因**：测试集中可能只有一个类别

**解决**：
1. 检查标签分布是否平衡
2. 调整test_size参数，确保测试集包含两个类别
3. 脚本会自动处理并给出警告

### Q4: 缺少xlrd模块？

**错误**：`ModuleNotFoundError: No module named 'xlrd'`

**解决**：
```powershell
pip install xlrd      # 用于读取.xls文件
pip install openpyxl  # 用于读取.xlsx文件
```

### Q5: Python版本兼容性问题？

**错误**：numpy、pandas等包导入失败

**解决**：
1. 使用Python 3.10（推荐）
2. 重新安装依赖包（参考环境要求部分）
3. 清除pip缓存：`pip cache purge`

## 模型特点

### 共同特点
- ✅ 使用随机森林分类器（效果稳定）
- ✅ 自动处理分类变量编码
- ✅ 自动处理缺失值
- ✅ 处理类别不平衡（使用class_weight='balanced'）
- ✅ 参数自动优化（RandomizedSearchCV）
- ✅ 生成ROC曲线和详细评估报告

### GEO模型特点
- 包含大量基因表达特征（数千个）
- 自动特征选择（选择前1000个重要特征）
- 适合高维数据

### STAD模型特点
- 主要使用临床特征（8-13个特征）
- 特征数较少，不进行特征选择
- 适合低维数据

## 注意事项

1. **数据准备**：确保数据文件在正确的目录下
2. **虚拟环境**：使用Python 3.10虚拟环境
3. **依赖包**：确保安装了所有必需的包
4. **类别不平衡**：两个数据集都存在类别不平衡问题，模型已自动处理
5. **R环境**：完整流程中的差异分析可以使用R的limma包（可选），如果R不可用，会自动使用Python的t-test

## 许可与作者

请根据需要添加 License 文件和作者信息。
