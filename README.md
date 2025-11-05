# 胃癌远处转移预测研究

本仓库用于基于 GSE15459 数据集的胃癌远处转移预测研究。包含多种预测模型实现、特征选择方法、生物信息学分析和R语言集成方案，旨在提供全面的胃癌预后预测和生物标志物发现平台。

## 项目概述

本研究旨在利用基因表达数据构建胃癌远处转移预测模型，通过多种机器学习方法和生物信息学分析技术，探索潜在的预后生物标志物，为胃癌患者的个体化治疗和预后评估提供科学依据。

### 主要研究内容：

1. **多维度特征选择**：结合LASSO回归、差异表达分析和WGCNA共表达网络分析，从全基因组表达谱中筛选关键预测特征。

2. **预测模型构建**：实现多种机器学习算法（随机森林、梯度提升、支持向量机、逻辑回归等）的预测模型，并进行模型集成优化。

3. **R语言生物信息学分析**：集成R语言强大的统计分析和可视化功能，进行基因功能注释（GO/KEGG分析）、表达谱差异分析等。

4. **WGCNA模块分析**：通过加权基因共表达网络分析识别关键基因模块，探索模块与胃癌转移的关联。

5. **预测性能评估**：使用交叉验证、准确率、精确率、召回率等多种指标全面评估模型性能。

## 目录结构

- **code/**
  - `generate_merged_data.py` —— 读取原始数据、清洗并生成 `data/data_merged.csv`（供训练使用）
  - `data_trans.py` —— 备用的数据转换脚本
  - `Prediction model of distant metastasis of stomach cancer.py` —— 基础预测模型实现
  - `integrated_prediction.py` —— 优化版集成预测模型，包含多种特征选择和集成分类器
  - `r_integration_prediction.py` —— R语言集成版预测模型，使用R的glmnet包进行LASSO特征选择
  - `R/` —— R语言脚本文件夹，包含特征选择和分析脚本
- **data/**（请把原始文件放在此处）
  - `GSE15459_series_matrix.txt` —— 基因表达数据
  - `GSE15459_outcome.xls` —— 样本结局数据
  - `data_merged.csv` —— 由 `generate_merged_data.py` 生成的合并数据集
- **output/** —— 模型预测结果和日志输出
- `.gitignore` —— Git忽略文件配置

## 三种预测模型的研究方法对比

### 1. 基础预测模型 (`Prediction model of distant metastasis of stomach cancer.py`)

**研究方法**：基于随机森林的单算法预测模型

**主要特点**：
- 采用随机森林算法进行胃癌远处转移风险预测
- 实现基本的特征筛选和模型评估流程
- 提供胃癌预后预测的基准方法

**研究价值**：
- 作为预测模型的基准对照组
- 初步探索基因表达与胃癌转移的关联
- 提供简单直观的预测模型框架

### 2. 优化版集成预测模型 (`integrated_prediction.py`)

**研究方法**：多特征选择方法与多模型集成的综合预测策略，结合高级可视化分析

**主要特点**：
- 结合LASSO回归、差异表达分析和WGCNA模块分析进行多维度特征筛选
- 构建基于特征重要性的权重系统，提高关键生物标志物的贡献
- 集成多种机器学习算法（随机森林、梯度提升、SVC、逻辑回归）
- 应用10折交叉验证确保模型稳定性和泛化能力
- **新增可视化功能**：提供ROC曲线、混淆矩阵、特征重要性排序等多种直观可视化结果
- **特征重要性分析**：详细分析并输出影响胃癌转移预测的关键生物标志物列表
- **模型性能比较**：可视化对比各分类器的预测性能和贡献

**研究价值**：
- 提高胃癌远处转移预测的准确性和稳定性
- 识别潜在的胃癌预后生物标志物集合
- 为临床个体化预后评估提供更可靠的模型支持

### 3. R语言集成预测模型 (`r_integration_prediction.py`)

**研究方法**：Python与R混合编程的生物信息学预测框架

**主要特点**：
- 利用R语言glmnet包进行专业的LASSO特征选择，适合高维基因表达数据
- 集成R的统计分析功能，提供更严格的差异表达基因筛选
- 结合Python的机器学习优势和R的生物信息学分析能力
- 支持与R生态系统中的GO/KEGG富集分析等功能无缝衔接

**研究价值**：
- 充分利用生物信息学专业工具进行特征筛选和功能分析
- 提供更全面的基因功能注释和通路分析
- 适合需要深入生物信息学分析的研究场景
- 为胃癌转移机制研究提供多组学数据分析支持

## 快速开始

1. 克隆仓库或在本地仓库根目录打开 PowerShell。

2. 把原始数据放到 data/ 下：
   - `GSE15459_series_matrix.txt`
   - `GSE15459_outcome.xls`

3. 在虚拟环境中安装依赖：
   ```
   pip install pandas scikit-learn xlrd openpyxl numpy scipy statsmodels rpy2
   ```
   注：使用 `r_integration_prediction.py` 时还需要安装R及相关包

4. 生成合并数据：
   ```
   python code\generate_merged_data.py
   ```
   生成文件路径：`data\data_merged.csv`

5. 运行预测模型：
   
   - 基础模型：
     ```
     python "code\Prediction model of distant metastasis of stomach cancer.py"
     ```
   
   - 优化版集成模型：
     ```
     python code\integrated_prediction.py
     ```
   
   - R语言集成模型：
     ```
     python code\r_integration_prediction.py
     ```

## 输出文件说明

- `output/predictions_integrated.csv` —— 优化版模型的预测结果
- `output/predictions_all_integrated.csv` —— 优化版模型的完整预测结果（包含概率）
- `output/predictions_r_integration.csv` —— R语言集成模型的预测结果
- `output/r_feature_selection_log.txt` —— R语言特征选择过程日志

### 可视化结果文件（`output/visualizations/`目录）

- `roc_curve.png` —— ROC曲线，展示模型的区分能力
- `precision_recall_curve.png` —— 精确率-召回率曲线，评估模型在不同阈值下的性能
- `confusion_matrix.png` —— 混淆矩阵可视化，直观展示分类结果
- `top_features.png` —— 前20个最重要特征的排序图
- `rf_feature_importance.png` —— 随机森林模型识别的关键特征
- `prediction_probability_distribution.png` —— 预测概率分布图，分析不同类别样本的概率分布
- `classifiers_performance.png` —— 各分类器性能比较图
- `feature_importance.csv` —— 前50个最重要特征的详细数据表格

## 注意事项

- 推荐不要将大数据文件和中间结果提交到仓库（已在 .gitignore 中配置）。
- 使用R集成功能需要正确配置R环境和相关包（如glmnet）。
- 可以根据具体需求调整各模型中的参数以获得更好的性能。

## 许可与作者

请根据需要添加 License 文件和作者信息。
