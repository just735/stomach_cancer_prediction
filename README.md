# Prediction-model-of-distant-metastasis-of-breast-cancer

项目：基于多源基因表达数据与机器学习的胃癌远处转移预测与生物标志物挖掘

本仓库包含用于构建、评估和汇总胃癌远处转移预测模型的脚本与结果，重点脚本为 `code/predict_rf.py`，该脚本实现了多源数据预处理（含简单回归批次效应去除）、候选随机种子筛选、随机森林模型搜索、阈值选择与结果保存；在有 GPU 且安装 PyTorch 时，会提供基于轻量 MLP 的可选训练路径。

---

**目录结构（核心）**
- `code/`：脚本目录，主要文件：`predict_rf.py`（重点）、`predict.py`、`collect_results.py`。
- `data/geodata.csv`：项目默认输入数据（或 `output/r_pipeline/clean_dataset.csv` 回退）。
- `output/`：所有运行结果均输出至 `output/r_pipeline/...` 或 `output/summary/`。

---

**快速开始（环境与依赖）**
- 建议使用 Python 3.9+ 虚拟环境。示例安装：

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib
# 可选（GPU/MLP）：
pip install torch
```

（如果你已有虚拟环境并已安装依赖，可跳过）

---

**输入数据格式要求**
- 文件：`data/geodata.csv`（或 `output/r_pipeline/clean_dataset.csv`）
- 列：必须包含 `sample_id` （可选）、`label`，以及多个基因表达列。
- `label` 值应为 `metastasis`（阳性）或 `control`（阴性）。
- 可选的批次列名（脚本自动检测）：`batch`, `Batch`, `batch_id`, `BatchID`, `dataset`, `study`, `platform`, `source`, `center`。

脚本会自动将非数值列转为数值（无法解析的值填为 0），并在检测到批次列时尝试使用回归方法去除批次效应。

---

**重点：`code/predict_rf.py` 使用说明**

- 功能概览：
  - 自动加载数据并尝试去除批次效应（回归逐基因法）。
  - 候选随机种子快速筛选（`rank_seeds`），选择表现稳定的若干种子用于后续搜索。
  - 在每个候选种子上进行模型搜索：默认使用 `RandomizedSearchCV` 对 `RandomForest` 管道进行超参数搜索；若检测到可用 GPU 且安装了 `torch`，将改为尝试使用轻量 `TorchMLPWrapper` 作为可选路径以利用 GPU 加速（小网格）。
  - 使用交叉验证概率输出选取最佳判别阈值，并在测试集合上评估 ACC、F1、AUC 等指标。
  - 保存模型、阈值、特征重要性（若适用）、混淆矩阵图、ROC/PR 曲线、预测结果与性能表。

- 运行示例：在项目根目录下执行：

```powershell
& .\.venv311\Scripts\Activate.ps1
python code/predict_rf.py
```

- 主要输出（每次运行会在 `output/r_pipeline/prediction_rf/<timestamp>/` 下生成一套结果）：
  - `model_performance.csv`：模型性能摘要（model, accuracy, f1, auc, seed, ...）。
  - `predictions_all.csv` / `predictions_test.csv`：样本级预测与概率。
  - `rf_model.pkl`（或 `gpu_model.pth`）：保存的模型（sklearn 使用 joblib，PyTorch 使用 state_dict）。
  - `feature_importance.csv` 与 `feature_importance_top30.png`：仅在 sklearn RF 路径可用。
  - `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`：可视化图像。
  - `best_params.json`、`best_threshold.json`、`run_summary.txt`：运行记录与参数。

- 常见可调整参数（在脚本顶部常量）：
  - `TEST_SIZE`：测试集占比（默认 0.1）。
  - `TARGET_ACC_MIN` / `TARGET_ACC_MAX`：阈值搜索目标区间。
  - 其他超参数在脚本内部的搜索空间中可修改以满足需要。

- 注意事项：
  - 若环境缺少 `scikit-learn` 等依赖，脚本会报错，请先安装对应包。
  - PyTorch 为可选依赖：若未安装或无 GPU，则使用 sklearn 的默认 CPU 路径。
  - 当前批次校正为回归逐基因方法，适合快速去批次验算；对更严格的生物学分析建议使用 ComBat（R 包 `sva`）或更复杂的批次校正流程。

---

**`code/predict.py`（简要）**
- 作用：对多种模型（LR、SVM、RF、ET、GBDT、LDA、QDA、KNN）进行网格搜索并比较性能；在有 GPU 时也会训练一个简易的 PyTorch MLP 作为备选。
- 输出：`output/r_pipeline/prediction/` 下的 `model_performance.csv`, `best_model_*.pkl/.pth`, `predictions.csv`, `predictions_all.csv`。

---

**`code/collect_results.py`（简要）**
- 作用：在工作区内搜索所有 `model_performance.csv`，聚合为 `output/summary/aggregated_model_performance.csv`，并生成汇总统计与可视化图表（如 mean ± std 条形图）。
- 使用示例：

```powershell
python code/collect_results.py --workspace . --out output/summary
```

---

**调试与常见问题**
- 报错 `ModuleNotFoundError: No module named 'sklearn'`：安装 `scikit-learn`。类似地需安装 `pandas', `numpy', `matplotlib` 等。
- 批次校正提示失败：脚本会捕获并记录该错误，流程会继续。若需更严谨的批次校正，请在 R 中使用 `ComBat` 后将清洗后的表达矩阵放置为 `data/geodata.csv`。

---

**复现实验与结果复盘**
- 所有运行会在 `output/r_pipeline/` 下生成时间戳目录，建议保留这些目录以便横向比对。已实现的聚合脚本会把多次实验结果合并到 `output/summary/` 以便统计分析。

---

如果你希望我：
- 把 `README.md` 转为 LaTeX/Word 或生成期刊格式的 Methods/Results 部分；
- 生成 `requirements.txt` 或 `environment.yml`；
- 为 `code/predict_rf.py` 增加命令行参数支持（例如指定输入文件、输出目录、GPU 强制或禁用开关），

请告诉我你想优先的下一步。
# 胃癌远处转移预测研究

本仓库用于基于多数据集的胃癌远处转移预测研究。包含完整的数据整合、预处理、特征筛选和模型训练流程，旨在提供全面的胃癌预后预测和生物标志物发现平台。

## 项目概述

本研究旨在利用基因表达数据和临床数据构建胃癌远处转移预测模型，通过多种机器学习方法和生物信息学分析技术，探索潜在的预后生物标志物，为胃癌患者的个体化治疗和预后评估提供科学依据。

## 系统介绍

本系统提供两条可并行的预测路径：一条面向多GEO表达谱整合的随机森林预测流程，另一条面向TCGA-STAD临床特征的快速预测流程。系统包含数据清洗、特征预处理、模型训练、阈值选择与结果输出，结果文件统一保存在 `result/` 与 `output/` 下，便于复现与评估。

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

## 实验结论与成果

### GEO表达谱随机森林实验（predict_rf.py）

- 最佳模型：随机森林（RF）
- 指标（测试集）：Accuracy 0.8571、F1 0.8571、AUC 0.9388
- 结果文件：output/r_pipeline/prediction_rf/model_performance.csv，output/r_pipeline/prediction_rf/predictions_all.csv

结论：GEO整合数据上的RF模型表现稳定，准确率与AUC均较高，阈值调节后保持目标区间准确率。

相关表达谱特征（Top10探针）：
226828_s_at、219463_at、1569003_at、1566690_at、204241_at、201506_at、227792_at、222795_s_at、225311_at、201308_s_at

### TCGA-STAD临床特征模型（STAD）

- 指标（测试集）：Accuracy 0.9773、F1 0.8333、AUC 0.9980
- 结果目录：output/STAD/20260211_210124/

结论：基于STAD临床特征的RF模型在AUC上达到极高水平，说明临床分期与转移相关字段具备强预测性；F1略低提示正负类不均衡下仍有改进空间。

相关临床因素（Top10重要特征）：
diagnoses.ajcc_pathologic_stage、diagnoses.residual_disease、diagnoses.days_to_last_follow_up、diagnoses.age_at_diagnosis、diagnoses.ajcc_pathologic_t、diagnoses.ajcc_pathologic_n、demographic.age_at_index、demographic.vital_status、diagnoses.tumor_grade、demographic.gender

### GEO多模型对比实验（predict.py）

- 最佳模型：RF
- 指标（测试集）：Accuracy 0.8571、F1 0.8571、AUC 0.9388
- 结果文件：output/r_pipeline/prediction/model_performance.csv，output/r_pipeline/prediction/predictions_all.csv

结论：多模型对比中RF在当前划分上取得最高准确率，AUC保持在较高水平，适合作为GEO流程的默认模型。

### 项目成果汇总

- 形成两类可复现的预测流程：GEO表达谱与STAD临床特征
- 输出可直接复用的模型与预测结果文件
- 统一的结果目录结构便于后续对比与融合建模

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
│   ├── R/                         # R分析脚本
│   ├── STAD/                      # STAD临床预测脚本
│   ├── build_geo_clean_dataset.py
│   ├── gastric_metastasis_pipeline.py
│   ├── gastric_metastasis_prediction_pipeline.R
│   ├── gastric_model_training.R
│   ├── predict.py
│   └── predict_rf.py
├── data/                          # 数据目录
│   ├── GEO/                       # GEO数据集
│   ├── STAD/                      # TCGA-STAD数据集
│   ├── STAD_processed/            # STAD处理后数据
│   ├── kaggle/                    # Kaggle表达谱
│   ├── processed_gastric/         # 处理后的表达谱数据
│   └── geodata.csv                # GEO整合数据
├── output/                        # 输出目录
│   ├── STAD/                      # STAD预测结果
│   ├── gastric/                   # 胃癌综合流程输出
│   └── r_pipeline/                # R/Python流程输出
├── result/                        # 固化结果与图表
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

- GEO多模型对比：`output/r_pipeline/prediction/`
- GEO随机森林：`output/r_pipeline/prediction_rf/`
- STAD临床模型：`output/STAD/`

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
