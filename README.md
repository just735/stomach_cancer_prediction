# Stomach cancer prediction

本仓库用于基于 GSE15459 数据的胃癌预后/转移预测示例。包含数据清洗脚本与模型训练脚本（随机森林示例）。

目录
- code/
  - generate_merged_data.py  —— 读取 raw data、清洗并生成 data/data_merged.csv（供训练使用）
  - Prediction model of distant metastasis of breast cancer.py  —— 基于 data_merged.csv 的训练/评估脚本
  - data_trans.py（可选） —— 备用的数据转换脚本
- data/（请把原始文件放在此处）
  - GSE15459_series_matrix.txt
  - GSE15459_outcome.xls
  - data_merged.csv（由 generate_merged_data.py 生成）
- output/（模型预测等输出）
- .gitignore

快速开始
1. 克隆仓库或在本地仓库根目录打开 PowerShell。
2. 把原始数据放到 data/ 下：
   - GSE15459_series_matrix.txt
   - GSE15459_outcome.xls
3. 在虚拟环境中安装依赖（示例）：
   ```
   python -m pip install -r requirements.txt
   ```
   如果没有 requirements.txt，至少安装 pandas、scikit-learn、xlrd/openpyxl:
   ```
   pip install pandas scikit-learn xlrd openpyxl
   ```
4. 生成合并数据：
   ```
   python code\generate_merged_data.py
   ```
   生成文件路径：`data\data_merged.csv`

5. 运行训练脚本（会读取 data\data_merged.csv）：
   ```
   python "code\Prediction model of distant metastasis of breast cancer.py"
   ```

注意
- 推荐不要将大数据文件和中间结果提交到仓库（已在 .gitignore 中配置）。
- 如果要提交模型或小结果文件，请把 output/ 中的具体文件从 .gitignore 中排除或手工添加。

许可与作者
- 请根据需要添加 License 文件和作者信息。
