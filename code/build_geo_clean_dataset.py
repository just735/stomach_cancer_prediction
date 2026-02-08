from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import gastric_metastasis_pipeline as gmp

GEO_DIR = BASE_DIR / "data" / "GEO"
OUT_DIR = BASE_DIR / "output" / "r_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "clean_dataset.csv"


def extract_ks_id(row):
    for val in row.values:
        if isinstance(val, str):
            m = re.search(r"(KS_\d+_T)", val)
            if m:
                return m.group(1)
    return None


def process_gse15459():
    expr_path = GEO_DIR / "GSE15459_series_matrix.txt"
    outcome_path = GEO_DIR / "GSE15459_outcome.xls"
    expr, _ = gmp.read_geo_series_matrix(expr_path)
    expr = gmp.ensure_numeric(expr)
    try:
        clinical = pd.read_excel(outcome_path, engine="xlrd")
    except Exception:
        clinical = pd.read_excel(outcome_path, engine="openpyxl")
    possible = [c for c in clinical.columns if "GSM" in str(c) or "sample" in str(c).lower()]
    if not possible:
        return None
    gsm_col = possible[0]
    clinical["metastasis"] = gmp.extract_label_from_clinical(clinical)
    label_df = clinical[[gsm_col, "metastasis"]].rename(columns={gsm_col: "sample_id"})
    expr_t = expr.T
    expr_t["sample_id"] = expr_t.index
    merged = expr_t.merge(label_df, on="sample_id", how="inner")
    merged = merged.set_index("sample_id")
    return merged


def process_gse26901():
    expr_path = GEO_DIR / "GSE26901_series_matrix.txt"
    clinical_path = GEO_DIR / "GSE26901_GC_KosinUniv_ClinicalInformation.txt"
    expr, meta = gmp.read_geo_series_matrix(expr_path)
    expr = gmp.ensure_numeric(expr)
    clinical = pd.read_csv(clinical_path, sep="\t")
    if "Array id" not in clinical.columns or "AJCC Stage" not in clinical.columns:
        return None
    clinical["AJCC Stage"] = clinical["AJCC Stage"].astype(str)
    clinical["metastasis"] = clinical["AJCC Stage"].apply(lambda x: 1 if x.strip() in {"4", "IV"} else 0)
    label_map = dict(zip(clinical["Array id"], clinical["metastasis"]))
    meta = meta.copy()
    meta["ks_id"] = meta.apply(extract_ks_id, axis=1)
    meta["metastasis"] = meta["ks_id"].map(label_map)
    expr_t = expr.T
    expr_t["metastasis"] = meta["metastasis"]
    return expr_t


def process_gse62254():
    expr_path = GEO_DIR / "GSE62254_series_matrix.txt"
    expr, meta = gmp.read_geo_series_matrix(expr_path)
    expr = gmp.ensure_numeric(expr)
    meta["metastasis"] = meta.apply(gmp.infer_metastasis_label_from_row, axis=1)
    expr_t = expr.T
    expr_t["metastasis"] = meta["metastasis"]
    return expr_t


def process_gse84437():
    expr_path = GEO_DIR / "GSE84437_series_matrix.txt"
    expr, meta = gmp.read_geo_series_matrix(expr_path)
    expr = gmp.ensure_numeric(expr)
    meta["metastasis"] = meta.apply(gmp.infer_metastasis_label_from_row, axis=1)
    expr_t = expr.T
    expr_t["metastasis"] = meta["metastasis"]
    return expr_t


def process_gse159929():
    expr_path = GEO_DIR / "GSE159929_series_matrix.txt"
    expr, meta = gmp.read_geo_series_matrix(expr_path)
    expr = gmp.ensure_numeric(expr)
    meta["metastasis"] = meta.apply(gmp.infer_metastasis_label_from_row, axis=1)
    expr_t = expr.T
    expr_t["metastasis"] = meta["metastasis"]
    return expr_t


def feature_type(expr_df):
    cols = expr_df.columns
    if any(str(c).startswith("ILMN_") for c in cols):
        return "ilmn"
    affy_like = sum(1 for c in cols if str(c).endswith("_at"))
    if affy_like >= max(10, int(len(cols) * 0.05)):
        return "affy"
    return "gene"


def integrate_labeled(datasets):
    labeled = [d for d in datasets if d["data"]["metastasis"].notna().sum() > 0]
    if not labeled:
        return None
    groups = {}
    for item in labeled:
        expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        t = feature_type(expr)
        groups.setdefault(t, []).append(item)
    def group_size(g):
        return sum(len(it["data"]) for it in g)
    best_group = max(groups.values(), key=group_size)
    gene_sets = []
    for item in best_group:
        expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        gene_sets.append(set(expr.columns))
    common_genes = set.intersection(*gene_sets) if gene_sets else set()
    if len(common_genes) == 0:
        best_single = max(best_group, key=lambda it: len(it["data"]))
        expr = best_single["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        merged = expr.copy()
        merged["metastasis"] = best_single["data"]["metastasis"]
        merged["dataset"] = best_single["data"]["dataset"]
        return merged
    combined = []
    for item in best_group:
        expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        expr = expr.loc[:, sorted(common_genes)]
        merged = expr.copy()
        merged["metastasis"] = item["data"]["metastasis"]
        merged["dataset"] = item["data"]["dataset"]
        combined.append(merged)
    return pd.concat(combined, axis=0)


def clean_and_save(df):
    df = df[df["metastasis"].notna()].copy()
    label = df["metastasis"].map({1: "metastasis", 0: "control"})
    expr = df.drop(columns=["metastasis", "dataset"], errors="ignore")
    expr = gmp.ensure_numeric(expr)
    zero_rate = (expr.isna() | (expr == 0)).mean(axis=0)
    expr = expr.loc[:, zero_rate < 0.5]
    if expr.shape[1] == 0:
        raise ValueError("清洗后无可用特征")
    vars_ = expr.var(axis=0, skipna=True)
    cutoff = vars_.quantile(0.25)
    expr = expr.loc[:, vars_ >= cutoff]
    expr = (expr - expr.mean(axis=0)) / expr.std(axis=0)
    expr = expr.replace([np.inf, -np.inf], np.nan).fillna(0)
    out = expr.copy()
    out["label"] = label.values
    out.insert(0, "sample_id", out.index.astype(str))
    out.to_csv(OUT_PATH, index=False)


def main():
    datasets = []
    items = [
        ("GSE15459", process_gse15459),
        ("GSE26901", process_gse26901),
        ("GSE62254", process_gse62254),
        ("GSE84437", process_gse84437),
        ("GSE159929", process_gse159929),
    ]
    for name, fn in items:
        try:
            data = fn()
        except Exception:
            data = None
        if data is None or data.shape[0] == 0:
            continue
        labels = data["metastasis"] if "metastasis" in data.columns else None
        expr = data.drop(columns=["metastasis"], errors="ignore")
        expr = gmp.ensure_numeric(expr)
        if expr.shape[1] == 0:
            continue
        data = expr.copy()
        if labels is not None:
            data["metastasis"] = labels
        data = gmp.process_dataset(data, name)
        datasets.append({"name": name, "data": data})
    combined = integrate_labeled(datasets)
    if combined is None or combined.shape[0] == 0:
        raise ValueError("没有可用于整合的 GEO 数据集")
    clean_and_save(combined)
    print(f"清洗后的 GEO 数据已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
