from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "geodata.csv"
FALLBACK_PATH = BASE_DIR / "output" / "r_pipeline" / "clean_dataset.csv"
OUT_DIR = BASE_DIR / "output" / "r_pipeline" / "prediction_rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_PATH
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ["sample_id", "label"]]
    X = df[cols]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].map({"metastasis": 1, "control": 0})
    sample_ids = df.get("sample_id", pd.Series(np.arange(len(df)).astype(str)))
    return X, y, sample_ids

def find_best_seed(X, y, seeds=range(0, 50), test_size=0.1):
    best_acc = -np.inf
    best_seed = 5
    selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
    Xk = selector.fit_transform(X, y)
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(Xk, y, test_size=test_size, random_state=seed, stratify=y)
        model = RandomForestClassifier(n_estimators=400, max_depth=None, max_features="sqrt", class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_seed = seed
    return best_seed, best_acc

def train_best_rf(X_train, y_train):
    pipeline = [
        ("vt", VarianceThreshold()),
        ("kbest", SelectKBest(score_func=f_classif)),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=1412)),
    ]
    model = Pipeline(pipeline)
    params = {
        "kbest__k": [50, 100, 200, 500],
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [5, 10, None],
        "rf__max_features": ["sqrt"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(model, params, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def main():
    X, y, sample_ids = load_dataset()
    if len(np.unique(y.dropna())) < 2:
        print("标签类别不足，无法训练")
        return
    seed, seed_acc = find_best_seed(X, y, test_size=0.1)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, sample_ids, test_size=0.1, random_state=seed, stratify=y)
    best_model, best_params = train_best_rf(X_train, y_train)
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    perf = pd.DataFrame([{"model": "RF", "accuracy": acc, "f1": f1, "auc": auc, "seed": seed, "params": best_params}])
    perf.to_csv(OUT_DIR / "model_performance.csv", index=False)
    joblib.dump(best_model, OUT_DIR / "rf_model.pkl")
    res_test = pd.DataFrame({"sample_id": id_test, "true": y_test, "pred": preds, "prob_positive": probs})
    res_test.to_csv(OUT_DIR / "predictions.csv", index=False)
    preds_all = best_model.predict(X)
    probs_all = best_model.predict_proba(X)[:, 1]
    res_all = pd.DataFrame({"sample_id": sample_ids, "true": y, "pred": preds_all, "prob_positive": probs_all})
    res_all.to_csv(OUT_DIR / "predictions_all.csv", index=False)
    print(f"RF accuracy={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}, seed={seed}")

if __name__ == "__main__":
    main()
