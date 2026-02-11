from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "STAD_processed" / "stad_clinical_processed.csv"
OUT_BASE = BASE_DIR / "output" / "STAD"
RUN_DIR = OUT_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_BASE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


def encode_features(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            codes, _ = pd.factorize(out[col].astype(str), sort=True)
            out[col] = codes
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise ValueError("数据缺少label列")
    sample_ids = df.get("sample_id", pd.Series(np.arange(len(df)).astype(str)))
    y = df["label"].map({1: 1, 0: 0, "1": 1, "0": 0})
    X = df.drop(columns=["label", "sample_id"], errors="ignore")
    X = encode_features(X)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    sample_ids = sample_ids.loc[valid]
    if y.nunique() < 2:
        print("标签类别不足，无法训练")
        return
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, sample_ids, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    params = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", 0.2, 0.3],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "bootstrap": [True, False],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model, params, cv=cv, scoring="roc_auc", n_iter=20, n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    joblib.dump(best_model, RUN_DIR / "rf_model.pkl")
    with open(RUN_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, ensure_ascii=False, indent=2)
    perf = pd.DataFrame([{"model": "RF", "accuracy": acc, "f1": f1, "auc": auc}])
    perf.to_csv(RUN_DIR / "model_performance.csv", index=False)
    fi = pd.DataFrame(
        {"feature": X.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi.to_csv(RUN_DIR / "feature_importance.csv", index=False)
    res_test = pd.DataFrame(
        {"sample_id": id_test, "true": y_test, "pred": preds, "prob_positive": probs}
    )
    res_test.to_csv(RUN_DIR / "predictions_test.csv", index=False)
    probs_all = best_model.predict_proba(X)[:, 1]
    preds_all = (probs_all >= 0.5).astype(int)
    res_all = pd.DataFrame(
        {"sample_id": sample_ids, "true": y, "pred": preds_all, "prob_positive": probs_all}
    )
    res_all.to_csv(RUN_DIR / "predictions_all.csv", index=False)
    print(f"STAD模型完成: acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")
    print(f"输出目录: {RUN_DIR}")


if __name__ == "__main__":
    main()
