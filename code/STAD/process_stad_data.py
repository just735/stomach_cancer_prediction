from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CLINICAL_DIR = BASE_DIR / "data" / "STAD" / "clinical.project-tcga-stad.2026-01-25"
SAMPLE_DIR = BASE_DIR / "data" / "STAD" / "biospecimen.project-tcga-stad.2026-01-25"
OUT_DIR = BASE_DIR / "data" / "STAD_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "stad_clinical_processed.csv"


def normalize_missing(df):
    df = df.replace({"'--": np.nan, "--": np.nan})
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str).str.strip()
            mask = s.str.lower().isin(["'--", "--", "not reported", "unknown", "na", "n/a", ""])
            df.loc[mask, col] = np.nan
    return df


def extract_label(row):
    candidates = [
        "diagnoses.metastasis_at_diagnosis",
        "diagnoses.ajcc_pathologic_m",
        "diagnoses.ajcc_clinical_m",
        "diagnoses.uicc_pathologic_m",
    ]
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).upper()
            if "M1" in val or "YES" in val or "METASTASIS" in val:
                return 1
            if "M0" in val or "NO" in val:
                return 0
    stage_cols = [
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_clinical_stage",
    ]
    for col in stage_cols:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).upper()
            if "IV" in val or "STAGE 4" in val or val.strip() == "4":
                return 1
            if any(x in val for x in ["I", "II", "III", "STAGE 1", "STAGE 2", "STAGE 3"]):
                return 0
    return np.nan


def first_non_null(series):
    non = series.dropna()
    if len(non) > 0:
        return non.iloc[0]
    return np.nan


def aggregate_by_case(df, key):
    return df.groupby(key, dropna=True).agg(first_non_null)


def build_sample_features():
    sample_path = SAMPLE_DIR / "sample.tsv"
    if not sample_path.exists():
        return None
    sample = pd.read_csv(sample_path, sep="\t", low_memory=False)
    sample = normalize_missing(sample)
    if "cases.submitter_id" not in sample.columns:
        return None
    def is_tumor(row):
        for col in ["samples.tissue_type", "samples.sample_type", "samples.tumor_descriptor"]:
            if col in row.index and pd.notna(row[col]):
                val = str(row[col]).lower()
                if "tumor" in val or "primary" in val:
                    return True
        return False
    sample["__is_tumor__"] = sample.apply(is_tumor, axis=1)
    if sample["__is_tumor__"].any():
        sample = sample[sample["__is_tumor__"]]
    sample = sample.drop(columns=["__is_tumor__"], errors="ignore")
    sample = aggregate_by_case(sample, "cases.submitter_id")
    return sample


def main():
    clinical_path = CLINICAL_DIR / "clinical.tsv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"未找到临床数据: {clinical_path}")
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    clinical = normalize_missing(clinical)
    if "cases.submitter_id" in clinical.columns:
        key = "cases.submitter_id"
    elif "cases.case_id" in clinical.columns:
        key = "cases.case_id"
    else:
        raise ValueError("临床数据缺少病例ID列")
    clinical["label"] = clinical.apply(extract_label, axis=1)
    clinical = aggregate_by_case(clinical, key)
    samples = build_sample_features()
    if samples is not None:
        clinical = clinical.join(samples, how="left", rsuffix="_sample")
    drop_cols = [c for c in clinical.columns if "metastasis" in c.lower() or c.lower().endswith("_m")]
    drop_cols = list(set(drop_cols) | {"label"})
    features = clinical.drop(columns=drop_cols, errors="ignore")
    label = clinical["label"]
    miss_rate = features.isna().mean()
    features = features.loc[:, miss_rate < 0.6]
    out = features.copy()
    out.insert(0, "sample_id", out.index.astype(str))
    out["label"] = label.values
    out.to_csv(OUT_PATH, index=False)
    print(f"STAD清洗完成，输出: {OUT_PATH}")
    print(f"样本数: {out.shape[0]}, 特征数: {out.shape[1] - 2}")


if __name__ == "__main__":
    main()
