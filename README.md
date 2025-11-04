# 乳腺癌远处转移预测模型

本仓库用于基于 GSE15459 数据集的乳腺癌远处转移预测研究。包含多种预测模型实现、特征选择方法和R语言集成方案，旨在提供全面的乳腺癌预后预测解决方案。

## 项目优化总结

### 已完成的主要改进：

1. **R语言集成修复**：修复了`r_integration_prediction.py`中LASSO特征选择时出现的"S4类下标无效"错误，通过添加`as.matrix()`将R的S4类对象转换为矩阵格式，确保特征选择正常运行。

2. **模型准确率比较**：对比分析了不同预测脚本的性能表现，`r_integration_prediction.py`准确率为0.6379，`integrated_prediction.py`优化后基础准确率达到0.6724。

3. **特征选择优化**：
   - 改进LASSO特征选择策略，使用10折交叉验证和动态特征数量调整
   - 增强差异表达分析方法，综合t统计量和效应量
   - 优化WGCNA模块特征提取，使用相关基因加权平均

4. **模型集成增强**：实现了多模型集成分类器，包括随机森林、梯度提升、SVC和逻辑回归的软投票集成，提供更稳定的预测性能。

5. **特征重要性权重系统**：添加了特征评分机制，根据不同方法的置信度为特征分配权重。

## 目录结构

- **code/**
  - `generate_merged_data.py` —— 读取原始数据、清洗并生成 `data/data_merged.csv`（供训练使用）
  - `data_trans.py` —— 备用的数据转换脚本
  - `Prediction model of distant metastasis of breast cancer.py` —— 基础预测模型实现
  - `integrated_prediction.py` —— 优化版集成预测模型，包含多种特征选择和集成分类器
  - `r_integration_prediction.py` —— R语言集成版预测模型，使用R的glmnet包进行LASSO特征选择
  - `R/` —— R语言脚本文件夹，包含特征选择和分析脚本
- **data/**（请把原始文件放在此处）
  - `GSE15459_series_matrix.txt` —— 基因表达数据
  - `GSE15459_outcome.xls` —— 样本结局数据
  - `data_merged.csv` —— 由 `generate_merged_data.py` 生成的合并数据集
- **output/** —— 模型预测结果和日志输出
- `.gitignore` —— Git忽略文件配置

## 三种预测模型的对比

### 1. 基础预测模型 (`Prediction model of distant metastasis of breast cancer.py`)

**预测方式**：单算法实现，使用随机森林作为基础分类器

**特点**：
- 实现简单直观，适合入门学习
- 使用基本的数据预处理和特征选择方法
- 提供基础的模型评估指标

**用途**：
- 作为基准模型进行性能比较
- 快速原型开发和验证
- 教学演示和学习参考

### 2. 优化版集成预测模型 (`integrated_prediction.py`)

**预测方式**：多特征选择方法 + 多模型集成

**特点**：
- 综合使用LASSO特征选择、差异表达分析和WGCNA模块特征提取
- 实现了特征重要性权重系统
- 集成随机森林、梯度提升、SVC和逻辑回归为投票分类器
- 使用10折交叉验证优化参数

**用途**：
- 追求更高预测准确性的场景
- 需要稳定预测结果的应用
- 特征重要性分析和生物学解释

### 3. R语言集成预测模型 (`r_integration_prediction.py`)

**预测方式**：Python与R混合编程，使用R的专业统计包

**特点**：
- 利用R语言强大的统计分析能力
- 使用R的glmnet包进行高级LASSO特征选择
- 集成R的统计检验和可视化功能
- 支持更多生物信息学分析方法

**用途**：
- 需要R语言专有功能的场景
- 与现有R分析流程集成
- 利用R生态系统中的专业统计方法
- 生物信息学相关分析和可视化

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
     python "code\Prediction model of distant metastasis of breast cancer.py"
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

## 注意事项

- 推荐不要将大数据文件和中间结果提交到仓库（已在 .gitignore 中配置）。
- 使用R集成功能需要正确配置R环境和相关包（如glmnet）。
- 可以根据具体需求调整各模型中的参数以获得更好的性能。

## 许可与作者

请根据需要添加 License 文件和作者信息。
