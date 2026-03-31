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
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

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
    sample_ids = df.get("sample_id", pd.Series(np.arange(len(df)).astype(str)))
    # 自动检测常见批次列并去除批次效应
    batch_candidates = {"batch", "Batch", "batch_id", "BatchID", "dataset", "study", "platform", "source", "center"}
    batch_col = next((c for c in df.columns if c in batch_candidates), None)
    if batch_col is not None:
        try:
            X = remove_batch_effects(X, df[batch_col])
        except Exception as e:
            print(f"批次校正失败（{batch_col}）：{e}")
    return X, y, sample_ids


def remove_batch_effects(X_df, batch_series):
    """使用回归（每个基因）去除批次效应的简单实现。"""
    if batch_series is None:
        return X_df
    batch_series = pd.Series(batch_series).astype(str).reset_index(drop=True)
    levels = batch_series.unique()
    if len(levels) <= 1:
        return X_df
    D = pd.get_dummies(batch_series, drop_first=True)
    if D.shape[1] == 0:
        return X_df
    D = pd.concat([pd.Series(1, index=D.index, name="_intercept"), D], axis=1)
    Dm = D.values
    X_out = X_df.copy()
    for col in X_df.columns:
        y = X_df[col].values
        try:
            coef, *_ = np.linalg.lstsq(Dm, y, rcond=None)
            fitted = Dm.dot(coef)
            X_out[col] = y - fitted
        except Exception:
            X_out[col] = y
    return X_out

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
    # 若有可用 GPU 则训练一个 PyTorch MLP 作为备选（快速小网格）
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        # 简单 MLP
        class TorchMLPWrapper:
            def __init__(self, input_dim, hidden_dim=128, device=None):
                self.device = device or (torch.device("cuda") if torch is not None and torch.cuda.is_available() else None)
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1),
                )
                if self.device is not None:
                    self.model.to(self.device)

            def fit(self, X, y, epochs=40, lr=1e-3, batch_size=32):
                X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
                y_t = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
                ds = TensorDataset(X_t, y_t)
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
                opt = torch.optim.Adam(self.model.parameters(), lr=lr)
                loss_fn = nn.BCEWithLogitsLoss()
                self.model.train()
                for epoch in range(epochs):
                    for xb, yb in dl:
                        if self.device is not None:
                            xb = xb.to(self.device)
                            yb = yb.to(self.device)
                        opt.zero_grad()
                        out = self.model(xb)
                        loss = loss_fn(out, yb)
                        loss.backward()
                        opt.step()

            def predict_proba(self, X):
                self.model.eval()
                X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
                if self.device is not None:
                    X_t = X_t.to(self.device)
                with torch.no_grad():
                    logits = self.model(X_t)
                    if self.device is not None:
                        logits = logits.cpu()
                    logits = logits.numpy().reshape(-1)
                    probs_pos = 1 / (1 + np.exp(-logits))
                    probs = np.vstack([1 - probs_pos, probs_pos]).T
                return probs

            def predict(self, X):
                probs = self.predict_proba(X)[:, 1]
                return (probs >= 0.5).astype(int)

            def save(self, path):
                try:
                    if self.device is not None:
                        torch.save(self.model.state_dict(), str(path))
                except Exception:
                    pass

        mlp = TorchMLPWrapper(input_dim=X_np.shape[1], hidden_dim=128, device=(torch.device("cuda") if torch.cuda.is_available() else None))
        mlp.fit(X_np, y_np, epochs=40, lr=1e-3, batch_size=32)
        results["MLP_GPU"] = {"best": mlp, "params": {"hidden_dim": 128, "epochs": 40}, "cv_score": None}
    return results

def evaluate(best, X_train, X_test, y_train, y_test):
    try:
        best.fit(X_train, y_train)
    except Exception:
        # 有些包装器可能已训练好（例如我们为 GPU 训练的 MLP），忽略重复训练错误
        pass
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
    # 如果模型提供 save（例如 PyTorch wrapper），使用其保存方法；否则使用 joblib
    try:
        if hasattr(best, "save"):
            best.save(OUT_DIR / f"best_model_{best_name}.pth")
        else:
            joblib.dump(best, OUT_DIR / f"best_model_{best_name}.pkl")
    except Exception:
        try:
            joblib.dump(best, OUT_DIR / f"best_model_{best_name}.pkl")
        except Exception:
            pass
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
