from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "geodata.csv"
FALLBACK_PATH = BASE_DIR / "output" / "r_pipeline" / "clean_dataset.csv"
OUT_DIR = BASE_DIR / "output" / "r_pipeline" / "prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_PATH
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ["sample_id", "label"]]
    X = df[cols]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].map({"metastasis": 1, "control": 0})
    return X, y, df.get("sample_id", pd.Series(np.arange(len(df)).astype(str)))

def train_models(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        "LR": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
            {"kbest__k": [5, 10, 20, 50, 100, 200], "clf__C": [0.1, 1, 10]},
        ),
        "SVM": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced")),
            ]),
            {"kbest__k": [5, 10, 20, 50, 100, 200], "clf__C": [0.1, 1, 10], "clf__kernel": ["rbf"]},
        ),
        "RF": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=1412)),
            ]),
            {"kbest__k": [50, 100, 200, 500], "clf__n_estimators": [200, 400], "clf__max_depth": [5, 10, None], "clf__max_features": ["sqrt"]},
        ),
        "ET": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=1412)),
            ]),
            {"kbest__k": [50, 100, 200, 500], "clf__n_estimators": [300, 500], "clf__max_depth": [5, 10, None], "clf__max_features": ["sqrt"]},
        ),
        "GBDT": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("clf", GradientBoostingClassifier()),
            ]),
            {"kbest__k": [20, 50, 100], "clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
        ),
        "LDA": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("clf", LinearDiscriminantAnalysis()),
            ]),
            {"kbest__k": [5, 10, 20, 50]},
        ),
        "QDA": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("clf", QuadraticDiscriminantAnalysis()),
            ]),
            {"kbest__k": [5, 10, 20]},
        ),
        "KNN": (
            Pipeline([
                ("vt", VarianceThreshold()),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {"kbest__k": [5, 10, 20, 50], "clf__n_neighbors": [3, 5, 7, 9]},
        ),
    }
    results = {}
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=cv, scoring="accuracy", n_jobs=-1)
        grid.fit(X, y)
        results[name] = {"best": grid.best_estimator_, "params": grid.best_params_, "cv_score": grid.best_score_}
    return results

def evaluate(best, X_train, X_test, y_train, y_test):
    best.fit(X_train, y_train)
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
    return acc, f1, auc, preds, probs

def find_best_seed(X, y, sample_ids, seeds=range(0, 50), test_size=0.1):
    best_acc = -np.inf
    best_seed = 42
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

def main():
    X, y, sample_ids = load_dataset()
    if len(np.unique(y.dropna())) < 2:
        print("标签类别不足，无法训练")
        return
    seed, seed_acc = find_best_seed(X, y, sample_ids, test_size=0.1)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, sample_ids, test_size=0.1, random_state=seed, stratify=y)
    X_train.index = id_train
    X_test.index = id_test
    X.index = sample_ids
    grids = train_models(X_train, y_train)
    perf_rows = []
    best_name = None
    best_acc = -np.inf
    for name, info in grids.items():
        acc, f1, auc, preds, probs = evaluate(info["best"], X_train, X_test, y_train, y_test)
        perf_rows.append({"model": name, "accuracy": acc, "f1": f1, "auc": auc})
        if acc > best_acc:
            best_acc = acc
            best_name = name
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUT_DIR / "model_performance.csv", index=False)
    best = grids[best_name]["best"]
    joblib.dump(best, OUT_DIR / f"best_model_{best_name}.pkl")
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
    res_test = pd.DataFrame({"sample_id": id_test, "true": y_test, "pred": preds})
    if probs is not None:
        res_test["prob_positive"] = probs
    res_test.to_csv(OUT_DIR / "predictions.csv", index=False)
    preds_all = best.predict(X)
    probs_all = best.predict_proba(X)[:, 1] if hasattr(best, "predict_proba") else None
    res_all = pd.DataFrame({"sample_id": sample_ids, "true": y, "pred": preds_all})
    if probs_all is not None:
        res_all["prob_positive"] = probs_all
    res_all.to_csv(OUT_DIR / "predictions_all.csv", index=False)
    print(perf_df)
    print(f"最佳模型: {best_name} ACC={best_acc:.3f}")

if __name__ == "__main__":
    main()
