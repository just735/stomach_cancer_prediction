import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind
import joblib
import warnings
import sys
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
GEO_DIR = DATA_DIR / "GEO"
KAGGLE_DIR = DATA_DIR / "kaggle"
STAD_DIR = DATA_DIR / "STAD" / "clinical.project-tcga-stad.2026-01-25"
OUTPUT_DIR = BASE_DIR / "output" / "gastric"
PROCESSED_DIR = DATA_DIR / "processed_gastric"
LOG_DIR = BASE_DIR / "output" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def read_geo_series_matrix(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    begin = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            begin = i
        if line.startswith("!series_matrix_table_end"):
            end = i
            break
    if begin is None or end is None:
        raise ValueError(f"{path} 未找到 series_matrix_table 区块")
    header_line = lines[begin + 1].rstrip("\n")
    header_parts = header_line.split("\t")
    sample_ids = header_parts[1:]
    nrows = end - begin - 2
    expr = pd.read_csv(path, sep="\t", skiprows=begin + 1, nrows=nrows, index_col=0)
    meta = parse_geo_metadata(lines[:begin + 1], sample_ids)
    return expr, meta

def parse_geo_metadata(lines, sample_ids):
    meta = pd.DataFrame(index=sample_ids)
    for line in lines:
        if not line.startswith("!Sample_"):
            continue
        parts = line.rstrip("\n").split("\t")
        key = parts[0].replace("!Sample_", "").lower()
        values = [v.strip('"') for v in parts[1:]]
        if len(values) != len(sample_ids):
            continue
        if key == "characteristics_ch1":
            for idx, raw in enumerate(values):
                if ":" in raw:
                    k, v = raw.split(":", 1)
                    col = k.strip().lower().replace(" ", "_")
                    val = v.strip()
                    if col in meta.columns and pd.notna(meta.loc[sample_ids[idx], col]):
                        existing = str(meta.loc[sample_ids[idx], col])
                        if val not in existing.split(";"):
                            meta.loc[sample_ids[idx], col] = existing + ";" + val
                    else:
                        meta.loc[sample_ids[idx], col] = val
                else:
                    col = "characteristics_ch1"
                    meta.loc[sample_ids[idx], col] = raw
        else:
            meta[key] = values
    return meta

def infer_metastasis_label_from_row(row):
    for col in row.index:
        val = row[col]
        if isinstance(val, str):
            low = val.lower()
            if "metastasis" in low or "distant" in low:
                if any(x in low for x in ["m1", "yes", "positive", "true"]):
                    return 1
                if any(x in low for x in ["m0", "no", "negative", "false"]):
                    return 0
    for col in row.index:
        if "m_stage" in col or col.endswith("_m") or "ajcc" in col:
            val = str(row[col]).upper()
            if "M1" in val:
                return 1
            if "M0" in val:
                return 0
    for col in row.index:
        if "stage" in col:
            val = str(row[col]).upper()
            if "IV" in val or "STAGE 4" in val or val.strip() == "4":
                return 1
            if any(x in val for x in ["I", "II", "III", "STAGE 1", "STAGE 2", "STAGE 3"]):
                return 0
    return np.nan

def extract_label_from_clinical(clinical):
    def extract_row(row):
        cols = [c for c in clinical.columns if "metastasis" in str(c).lower()]
        for col in cols:
            val = str(row.get(col)).lower()
            if any(x in val for x in ["yes", "1", "metastasis", "positive", "true"]):
                return 1
            if any(x in val for x in ["no", "0", "none", "negative", "false"]):
                return 0
        cols = [c for c in clinical.columns if "outcome" in str(c).lower()]
        for col in cols:
            val = str(row.get(col)).lower()
            if "1" in val or "dead" in val:
                return 1
            if "0" in val or "alive" in val:
                return 0
        cols = [c for c in clinical.columns if "stage" in str(c).lower()]
        for col in cols:
            val = str(row.get(col)).upper()
            if "IV" in val or "4" in val:
                return 1
            if any(x in val for x in ["I", "II", "III"]):
                return 0
        return np.nan
    return clinical.apply(extract_row, axis=1)

def ensure_numeric(df):
    if df.shape[1] == 0:
        return df
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=0, how="all")
    return out

def collapse_probes(expr, mapping_path):
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return expr
    mapping = pd.read_csv(mapping_path)
    cols = [c.lower() for c in mapping.columns]
    mapping.columns = cols
    probe_col = None
    gene_col = None
    for c in cols:
        if "probe" in c or "id" == c or "id_ref" == c:
            probe_col = c
        if "gene" in c and ("symbol" in c or "name" in c):
            gene_col = c
    if probe_col is None or gene_col is None:
        return expr
    mapping = mapping[[probe_col, gene_col]].dropna()
    mapping = mapping.drop_duplicates(subset=[probe_col])
    expr = expr.copy()
    expr["__probe__"] = expr.index
    merged = expr.merge(mapping, left_on="__probe__", right_on=probe_col, how="inner")
    merged = merged.drop(columns=["__probe__", probe_col])
    merged = merged.set_index(gene_col)
    merged = merged.groupby(merged.index).mean()
    return merged

def remove_outliers(expr_samples, dataset_name, cut_height=150):
    if expr_samples.shape[0] < 10:
        return expr_samples, []
    expr_numeric = ensure_numeric(expr_samples)
    expr_numeric = expr_numeric.fillna(expr_numeric.median())
    if expr_numeric.shape[1] > 1000:
        variances = expr_numeric.var(axis=0)
        top_genes = variances.nlargest(int(expr_numeric.shape[1] * 0.25)).index
        cluster_data = expr_numeric[top_genes]
    else:
        cluster_data = expr_numeric
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(cluster_data), index=cluster_data.index, columns=cluster_data.columns)
    distance_matrix = pdist(scaled.values, metric="euclidean")
    linkage_matrix = linkage(distance_matrix, method="ward")
    clusters = fcluster(linkage_matrix, t=cut_height, criterion="distance")
    cluster_counts = pd.Series(clusters).value_counts()
    outlier_clusters = cluster_counts[cluster_counts <= 2].index
    outliers = []
    for cluster_id in outlier_clusters:
        outliers.extend(scaled.index[clusters == cluster_id].tolist())
    cleaned = expr_samples.drop(index=outliers, errors="ignore")
    print(f"{dataset_name} 异常样本数: {len(outliers)}")
    return cleaned, outliers

def zscore_by_gene(expr_genes):
    expr_numeric = ensure_numeric(expr_genes)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(expr_numeric.T)
    z = pd.DataFrame(scaled.T, index=expr_numeric.index, columns=expr_numeric.columns)
    return z

def low_variance_filter(expr_genes, min_variance=0.01):
    variances = expr_genes.var(axis=1)
    keep = variances[variances >= min_variance].index
    return expr_genes.loc[keep]

def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.shape[0]
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    qvals = np.empty(n, dtype=float)
    qvals[order] = pvals[order] * n / ranks
    qvals = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(qvals, 0, 1)
    return out

def deg_filter(expr_samples, labels, logfc_thresh=0.5, p_thresh=0.05):
    expr = expr_samples.T
    group1 = labels[labels == 1].index
    group0 = labels[labels == 0].index
    expr1 = expr[group1]
    expr0 = expr[group0]
    mean1 = expr1.mean(axis=1)
    mean0 = expr0.mean(axis=1)
    logfc = mean1 - mean0
    stat, pvals = ttest_ind(expr1.values, expr0.values, axis=1, equal_var=False, nan_policy="omit")
    adj = bh_adjust(pvals)
    genes = expr.index[(np.abs(logfc) > logfc_thresh) & (adj < p_thresh)]
    return genes

def wgcna_filter(expr_samples, labels, power=6, min_module_size=30, merge_cut_height=0.5, max_genes=5000):
    expr = expr_samples.T
    variances = expr.var(axis=1).sort_values(ascending=False)
    if len(variances) > max_genes:
        expr = expr.loc[variances.index[:max_genes]]
    corr = np.corrcoef(expr.values)
    np.fill_diagonal(corr, 1.0)
    dist = 1 - np.abs(corr)
    tri = dist[np.triu_indices_from(dist, k=1)]
    linkage_matrix = linkage(tri, method="average")
    clusters = fcluster(linkage_matrix, t=merge_cut_height, criterion="distance")
    modules = {}
    for gene, cid in zip(expr.index, clusters):
        modules.setdefault(cid, []).append(gene)
    modules = {k: v for k, v in modules.items() if len(v) >= min_module_size}
    if not modules:
        return expr.index
    labels_vec = labels.loc[expr.columns].values.astype(float)
    module_scores = []
    for cid, genes in modules.items():
        eigengene = expr.loc[genes].mean(axis=0).values
        if np.std(eigengene) == 0:
            corr_val = 0
        else:
            corr_val = np.corrcoef(eigengene, labels_vec)[0, 1]
        module_scores.append((cid, abs(corr_val)))
    module_scores.sort(key=lambda x: x[1], reverse=True)
    top_modules = [cid for cid, _ in module_scores[:2]]
    selected = []
    for cid in top_modules:
        selected.extend(modules[cid])
    return pd.Index(selected).unique()

def lasso_feature_selection(expr_samples, labels, candidate_genes):
    X = expr_samples[candidate_genes]
    X = ensure_numeric(X).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, labels.values.astype(float))
    coef = lasso.coef_
    selected = np.array(candidate_genes)[coef != 0]
    if len(selected) == 0:
        idx = np.argsort(np.abs(coef))[-min(21, len(candidate_genes)):]
        selected = np.array(candidate_genes)[idx]
    return pd.Index(selected)

def select_markers(expr_samples, labels):
    deg_genes = deg_filter(expr_samples, labels)
    wgcna_genes = wgcna_filter(expr_samples, labels)
    candidate_genes = pd.Index(deg_genes).intersection(pd.Index(wgcna_genes))
    if len(candidate_genes) == 0:
        candidate_genes = pd.Index(expr_samples.columns)
    marker_genes = lasso_feature_selection(expr_samples, labels, candidate_genes)
    marker_expr = expr_samples[marker_genes]
    return marker_genes, marker_expr

def load_tcga_label_map():
    clinical_path = STAD_DIR / "clinical.tsv"
    if not clinical_path.exists():
        return {}
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    def extract(row):
        for col in ["diagnoses.metastasis_at_diagnosis", "diagnoses.ajcc_pathologic_m", "diagnoses.ajcc_clinical_m", "diagnoses.uicc_pathologic_m"]:
            if col in clinical.columns and pd.notna(row.get(col)):
                val = str(row[col]).upper()
                if "M1" in val or "YES" in val or "METASTASIS" in val:
                    return 1
                if "M0" in val or "NO" in val:
                    return 0
        return np.nan
    clinical["label"] = clinical.apply(extract, axis=1)
    if "cases.submitter_id" in clinical.columns:
        ids = clinical["cases.submitter_id"]
    elif "cases.case_id" in clinical.columns:
        ids = clinical["cases.case_id"]
    else:
        return {}
    mapping = pd.Series(clinical["label"].values, index=ids).dropna()
    return mapping.to_dict()

def process_gse15459():
    expr_path = GEO_DIR / "GSE15459_series_matrix.txt"
    outcome_path = GEO_DIR / "GSE15459_outcome.xls"
    expr, _ = read_geo_series_matrix(expr_path)
    expr = ensure_numeric(expr)
    try:
        clinical = pd.read_excel(outcome_path, engine="xlrd")
    except Exception:
        clinical = pd.read_excel(outcome_path, engine="openpyxl")
    possible = [c for c in clinical.columns if "GSM" in str(c) or "sample" in str(c).lower()]
    if not possible:
        raise ValueError("GSE15459 临床表未找到样本列")
    gsm_col = possible[0]
    clinical["metastasis"] = extract_label_from_clinical(clinical)
    label_df = clinical[[gsm_col, "metastasis"]].rename(columns={gsm_col: "sample_id"})
    expr_t = expr.T
    expr_t["sample_id"] = expr_t.index
    merged = expr_t.merge(label_df, on="sample_id", how="inner")
    merged = merged.set_index("sample_id")
    return merged

def process_gse62254():
    expr_path = GEO_DIR / "GSE62254_series_matrix.txt"
    expr, meta = read_geo_series_matrix(expr_path)
    expr = ensure_numeric(expr)
    meta["metastasis"] = meta.apply(infer_metastasis_label_from_row, axis=1)
    expr_t = expr.T
    expr_t["metastasis"] = meta["metastasis"]
    return expr_t

def process_gse84437():
    path = GEO_DIR / "GSE84437_sample.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df = df[~df.index.str.startswith("Unnamed", na=False)]
    df = ensure_numeric(df)
    df["metastasis"] = np.nan
    return df

def process_kaggle(tcga_label_map):
    path = KAGGLE_DIR / "expression profile(8863 genes).csv"
    expr = pd.read_csv(path, index_col=0)
    expr = ensure_numeric(expr)
    expr_t = expr.T
    def map_label(sample_id):
        parts = str(sample_id).split("-")
        if len(parts) >= 3:
            base = "-".join(parts[:3])
            if base in tcga_label_map:
                return tcga_label_map[base]
        if sample_id in tcga_label_map:
            return tcga_label_map[sample_id]
        return np.nan
    expr_t["metastasis"] = expr_t.index.map(map_label)
    return expr_t

def process_dataset(expr_samples, dataset_name):
    expr_samples = expr_samples.copy()
    labels = expr_samples["metastasis"] if "metastasis" in expr_samples.columns else pd.Series(np.nan, index=expr_samples.index)
    expr_samples = expr_samples.drop(columns=["metastasis"], errors="ignore")
    expr_samples = ensure_numeric(expr_samples)
    expr_samples = expr_samples.fillna(expr_samples.median())
    expr_samples, _ = remove_outliers(expr_samples, dataset_name)
    expr_genes = expr_samples.T
    expr_genes = zscore_by_gene(expr_genes)
    expr_genes = low_variance_filter(expr_genes)
    expr_samples = expr_genes.T
    expr_samples["metastasis"] = labels.loc[expr_samples.index]
    expr_samples["dataset"] = dataset_name
    return expr_samples

def save_dataset(df, name):
    expr = df.drop(columns=["metastasis", "dataset"], errors="ignore")
    expr_path = PROCESSED_DIR / f"{name}_expr.csv"
    meta_path = PROCESSED_DIR / f"{name}_meta.csv"
    df.to_csv(PROCESSED_DIR / f"{name}_processed.csv")
    expr.to_csv(expr_path)
    meta = df[["metastasis", "dataset"]]
    meta.to_csv(meta_path)
    return expr_path

def integrate_labeled(datasets):
    labeled = [d for d in datasets if d["data"]["metastasis"].notna().sum() > 0]
    if not labeled:
        return None
    def feature_type(expr_df):
        cols = expr_df.columns
        if any(str(c).startswith("ILMN_") for c in cols):
            return "ilmn"
        affy_like = sum(1 for c in cols if str(c).endswith("_at"))
        if affy_like >= max(10, int(len(cols) * 0.05)):
            return "affy"
        return "gene"
    groups = {}
    for item in labeled:
        expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        t = feature_type(expr)
        groups.setdefault(t, []).append(item)
    # 选择样本数最多的组进行整合（通常为affy）
    def group_size(g):
        return sum(len(it["data"]) for it in g)
    best_type = None
    best_group = []
    for t, g in groups.items():
        if best_type is None or group_size(g) > group_size(best_group):
            best_type = t
            best_group = g
    if not best_group:
        return None
    gene_sets = []
    for item in best_group:
        expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        gene_sets.append(set(expr.columns))
    common_genes = set.intersection(*gene_sets) if gene_sets else set()
    if len(common_genes) == 0:
        # 回退：仅使用样本数最多的单一数据集，避免空特征
        best_single = max(best_group, key=lambda it: len(it["data"]))
        expr = best_single["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
        merged = expr.copy()
        merged["metastasis"] = best_single["data"]["metastasis"]
        merged["dataset"] = best_single["data"]["dataset"]
        combined_df = merged
    else:
        combined = []
        for item in best_group:
            expr = item["data"].drop(columns=["metastasis", "dataset"], errors="ignore")
            expr = expr.loc[:, sorted(common_genes)]
            merged = expr.copy()
            merged["metastasis"] = item["data"]["metastasis"]
            merged["dataset"] = item["data"]["dataset"]
            combined.append(merged)
        combined_df = pd.concat(combined, axis=0)
    out_path = PROCESSED_DIR / "integrated_labeled.csv"
    combined_df.to_csv(out_path)
    return out_path

def train_and_predict(marker_expr, labels):
    X = ensure_numeric(marker_expr).fillna(0)
    y = labels.loc[X.index]
    if len(y.unique()) < 2:
        print("标签类别不足，无法训练")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "LR": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
        "RF": (RandomForestClassifier(class_weight="balanced", random_state=42), {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}),
        "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
        "GBDT": (GradientBoostingClassifier(), {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}),
    }
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        probs = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
        results[name] = {"best_params": grid.best_params_, "accuracy": acc, "f1_score": f1, "auc": auc, "model": best}
        print(f"{name} - 准确率：{acc:.3f}, F1：{f1:.3f}, AUC：{auc:.3f}")
    best_model_name = max(results.keys(), key=lambda x: results[x]["auc"])
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, OUTPUT_DIR / "best_model.pkl")
    results_df = pd.DataFrame({
        "模型": results.keys(),
        "准确率": [v["accuracy"] for v in results.values()],
        "F1分数": [v["f1_score"] for v in results.values()],
        "AUC": [v["auc"] for v in results.values()],
    })
    results_df.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    results_test = pd.DataFrame({"true": y_test, "pred": preds}, index=y_test.index)
    if probs is not None:
        results_test["prob_positive"] = probs
    results_test.to_csv(OUTPUT_DIR / "predictions.csv")
    preds_all = best_model.predict(X)
    probs_all = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else None
    results_all = pd.DataFrame({"true": y, "pred": preds_all}, index=X.index)
    if probs_all is not None:
        results_all["prob_positive"] = probs_all
    results_all.to_csv(OUTPUT_DIR / "predictions_all.csv")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"gastric_pipeline_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    print("开始处理数据集")
    try:
        tcga_label_map = load_tcga_label_map()
        datasets = []
        gse15459 = process_gse15459()
        gse15459 = process_dataset(gse15459, "GSE15459")
        save_dataset(gse15459, "GSE15459")
        datasets.append({"name": "GSE15459", "data": gse15459})
        gse62254 = process_gse62254()
        gse62254 = process_dataset(gse62254, "GSE62254")
        save_dataset(gse62254, "GSE62254")
        datasets.append({"name": "GSE62254", "data": gse62254})
        gse84437 = process_gse84437()
        gse84437 = process_dataset(gse84437, "GSE84437")
        save_dataset(gse84437, "GSE84437")
        datasets.append({"name": "GSE84437", "data": gse84437})
        kaggle = process_kaggle(tcga_label_map)
        kaggle = process_dataset(kaggle, "Kaggle")
        save_dataset(kaggle, "Kaggle")
        datasets.append({"name": "Kaggle", "data": kaggle})
        integrated_path = integrate_labeled(datasets)
        if integrated_path is None:
            print("没有足够的带标签数据集用于训练")
            return
        print(f"整合数据已生成: {integrated_path}")
        data = pd.read_csv(integrated_path, index_col=0)
        data = data[data["metastasis"].notna()].copy()
        labels = data["metastasis"]
        expr = data.drop(columns=["metastasis", "dataset"], errors="ignore")
        marker_genes, marker_expr = select_markers(expr, labels)
        marker_expr.to_csv(PROCESSED_DIR / "marker_expr.csv")
        pd.Series(marker_genes).to_csv(PROCESSED_DIR / "marker_genes.csv", index=False)
        train_and_predict(marker_expr, labels)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

if __name__ == "__main__":
    main()
