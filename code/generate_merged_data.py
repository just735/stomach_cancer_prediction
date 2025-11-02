import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_EXPR = DATA_DIR / "GSE15459_series_matrix.txt"
DEFAULT_CLIN = DATA_DIR / "GSE15459_outcome.xls"
OUT_CSV = DATA_DIR / "data_merged.csv"

def fail(msg):
    print(msg); sys.exit(1)

expr_path = DEFAULT_EXPR
clin_path = DEFAULT_CLIN
out_path = OUT_CSV

if not expr_path.exists():
    fail(f"表达文件未找到: {expr_path}")
if not clin_path.exists():
    fail(f"临床文件未找到: {clin_path}")

# 读取表达矩阵（跳过以 ! 开头的注释行）
try:
    expr = pd.read_csv(expr_path, sep="\t", comment="!", index_col=0, engine="python", on_bad_lines="skip")
except Exception as e:
    fail(f"读取表达矩阵失败: {e}")

expr_t = expr.T.reset_index().rename(columns={"index": "sample_id"})

# 可选：排除已知不合格样本（若需要可修改列表）
excluded = ["GSM387788", "GSM387790", "GSM387793", "GSM387797",
            "GSM387798", "GSM387799", "GSM387842"]
expr_clean = expr_t[~expr_t["sample_id"].isin(excluded)].copy()

# 读取临床表（根据后缀选择 engine）
try:
    suf = clin_path.suffix.lower()
    if suf == ".xls":
        clinical = pd.read_excel(clin_path, engine="xlrd")
    elif suf == ".xlsx":
        clinical = pd.read_excel(clin_path, engine="openpyxl")
    else:
        clinical = pd.read_csv(clin_path, sep="\t", engine="python")
except Exception as e:
    fail(f"读取临床信息失败: {e}\n请安装相应依赖（xlrd/openpyxl）或检查文件格式。")

# 找到样本 ID 列
possible = [c for c in clinical.columns if "GSM" in str(c) or "sample" in str(c).lower()]
if not possible:
    fail("临床表中找不到包含 GSM 的列，请确认列名。")
gsm_col = possible[0]

# 选择核心临床列（若列不存在则以可用列替代或填充）
cols_to_pick = [gsm_col, "Age_at_surgery", "Gender", "Laurenclassification",
                "Stage", "Overall.Survival (Months)**", "Outcome (1=dead)"]
exist_cols = [c for c in cols_to_pick if c in clinical.columns]
clinical_core = clinical[exist_cols].rename(columns={gsm_col: "sample_id"})

# 对分类列做编码
for col in ["Gender", "Laurenclassification"]:
    if col in clinical_core.columns:
        clinical_core[col] = clinical_core[col].fillna("NA").astype(str)
        clinical_core[col] = LabelEncoder().fit_transform(clinical_core[col])
    else:
        clinical_core[col] = 0

# 合并
merged = pd.merge(expr_clean, clinical_core, on="sample_id", how="inner")
if merged.shape[0] == 0:
    fail("合并后无样本，检查 sample_id/GSM 是否匹配。")

# 将 sample_id 设为索引，标签（若存在）放到最后
merged = merged.set_index("sample_id", drop=True)
label = "Outcome (1=dead)"
if label in merged.columns:
    cols = [c for c in merged.columns if c != label] + [label]
    merged = merged[cols]

out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_path, index=True)
print(f"已生成: {out_path}  样本数: {merged.shape[0]}  列数: {merged.shape[1]}")
print("提示：后续训练脚本请使用 pd.read_csv(path, index_col=0) 读取此文件。")