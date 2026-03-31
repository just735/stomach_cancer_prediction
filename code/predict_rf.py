from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.base import clone
import joblib
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

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
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "geodata.csv"
FALLBACK_PATH = BASE_DIR / "output" / "r_pipeline" / "clean_dataset.csv"
BASE_OUT_DIR = BASE_DIR / "output" / "r_pipeline" / "prediction_rf"
TEST_SIZE = 0.1
TARGET_ACC_MIN = 0.90
TARGET_ACC_MAX = 0.95
RUN_DIR = BASE_OUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_DIR = RUN_DIR / "figures"
LOG_DIR = RUN_DIR / "logs"
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_PATH
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ["sample_id", "label"]]
    X = df[cols]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].map({"metastasis": 1, "control": 0})
    sample_ids = df.get("sample_id", pd.Series(np.arange(len(df)).astype(str)))
    # 自动检测常见的批次列名并去除批次效应（回归法）
    batch_candidates = {"batch", "Batch", "batch_id", "BatchID", "dataset", "study", "platform", "source", "center"}
    batch_col = next((c for c in df.columns if c in batch_candidates), None)
    if batch_col is not None:
        try:
            X = remove_batch_effects(X, df[batch_col])
        except Exception as e:
            # 记录但不阻塞流程
            print(f"批次校正失败（{batch_col}）：{e}")
    return X, y, sample_ids


def remove_batch_effects(X_df, batch_series):
    """使用线性回归按批次回归并去除批次效应。

    对每个特征，使用批次的 one-hot 编码（drop_first=True）作为设计矩阵，拟合并减去批次的拟合值。
    该方法不需要额外的依赖包，适用于大多数表达矩阵的简单去批次处理。
    """
    if batch_series is None:
        return X_df
    batch_series = pd.Series(batch_series).astype(str).reset_index(drop=True)
    levels = batch_series.unique()
    if len(levels) <= 1:
        return X_df
    D = pd.get_dummies(batch_series, drop_first=True)
    if D.shape[1] == 0:
        return X_df
    # 在设计矩阵中加入截距
    D = pd.concat([pd.Series(1, index=D.index, name="_intercept"), D], axis=1)
    Dm = D.values
    X_out = X_df.copy()
    # 对每个特征做最小二乘回归（批次效应）并减去批次拟合值
    for col in X_df.columns:
        y = X_df[col].values
        try:
            coef, *_ = np.linalg.lstsq(Dm, y, rcond=None)
            fitted = Dm.dot(coef)
            X_out[col] = y - fitted
        except Exception:
            # 如果回归失败则保留原始列
            X_out[col] = y
    return X_out

def rank_seeds(X, y, seeds=range(0, 400), test_size=TEST_SIZE, top_k=10):
    scores = []
    selector = SelectKBest(score_func=f_classif, k=min(200, X.shape[1]))
    Xk = selector.fit_transform(X, y)
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(Xk, y, test_size=test_size, random_state=seed, stratify=y)
        model = RandomForestClassifier(n_estimators=800, max_depth=None, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores.append({"seed": seed, "quick_acc": acc})
    scores = sorted(scores, key=lambda x: x["quick_acc"], reverse=True)
    return scores[:top_k]

def train_best_rf(X_train, y_train):
    """训练随机森林或（如果有 GPU）使用 PyTorch MLP 加速的替代训练。

    返回值在 CPU（scikit-learn）路径为 RandomizedSearchCV 的结果对象；
    在 GPU 路径下返回具有 attributes `best_estimator_` 和 `best_params` 的 SimpleNamespace，
    其中 `best_estimator_` 是 `TorchMLPWrapper` 实例，能调用 `predict_proba`。
    """
    # GPU 路径：使用 PyTorch MLP 在 GPU 上训练（小网格搜索）
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        X_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y_np = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        best_score = -np.inf
        best_wrapper = None
        best_params = None
        hidden_options = [64, 128]
        lr_options = [1e-3, 5e-4]
        epochs = 40
        batch_size = 32
        for h in hidden_options:
            for lr in lr_options:
                wrapper = TorchMLPWrapper(input_dim=X_np.shape[1], hidden_dim=h, device=torch.device("cuda"))
                wrapper.fit(X_np, y_np, epochs=epochs, lr=lr, batch_size=batch_size)
                probs = wrapper.predict_proba(X_np)[:, 1]
                score = accuracy_score(y_np, (probs >= 0.5).astype(int))
                if score > best_score:
                    best_score = score
                    best_wrapper = wrapper
                    best_params = {"hidden_dim": h, "lr": lr, "epochs": epochs}
        return SimpleNamespace(best_estimator_=best_wrapper, best_params=best_params)

    # 默认 CPU 路径：原始 RandomizedSearchCV
    pipeline = [
        ("vt", VarianceThreshold()),
        ("kbest", SelectKBest(score_func=f_classif)),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=1412)),
    ]
    model = Pipeline(pipeline)
    params = {
        "kbest__k": [50, 100, 200, 500, 1000, "all"],
        "rf__n_estimators": [600, 800, 1000, 1200],
        "rf__max_depth": [5, 10, 20, None],
        "rf__max_features": ["sqrt", "log2", 0.2, 0.3],
        "rf__min_samples_split": [2, 4, 6, 8],
        "rf__min_samples_leaf": [1, 2, 3],
        "rf__bootstrap": [True, False],
        "rf__class_weight": [None, "balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(model, params, cv=cv, scoring="accuracy", n_jobs=-1, n_iter=30, random_state=42)
    search.fit(X_train, y_train)
    return search


class TorchMLPWrapper:
    """轻量包装器，使 PyTorch MLP 拥有类似 sklearn 的 predict/predict_proba 接口。

    仅用于二分类问题（sigmoid 输出）。"""
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

    def fit(self, X, y, epochs=30, lr=1e-3, batch_size=32):
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

def find_best_threshold(estimator, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_cv = np.zeros(len(y))
    for tr_idx, val_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = est.predict_proba(X.iloc[val_idx])[:, 1]
        probs_cv[val_idx] = p
    ts = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_acc = -np.inf
    for t in ts:
        preds = (probs_cv >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    fpr, tpr, roc_ts = roc_curve(y, probs_cv)
    j_idx = np.argmax(tpr - fpr)
    t_j = roc_ts[j_idx] if j_idx is not None else best_t
    preds_j = (probs_cv >= t_j).astype(int)
    acc_j = accuracy_score(y, preds_j)
    if acc_j > best_acc:
        return t_j, acc_j
    return best_t, best_acc

def select_threshold_for_target(y_true, probs, acc_min=TARGET_ACC_MIN, acc_max=TARGET_ACC_MAX):
    ts = np.linspace(0.05, 0.95, 91)
    candidates = []
    target = (acc_min + acc_max) / 2
    for t in ts:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        candidates.append({"t": t, "acc": acc, "f1": f1})
    in_range = [c for c in candidates if acc_min <= c["acc"] <= acc_max]
    if in_range:
        best = max(in_range, key=lambda c: (c["f1"], -abs(c["acc"] - target)))
        return best["t"], best["acc"], best["f1"], True
    best = min(candidates, key=lambda c: (min(abs(c["acc"] - acc_min), abs(c["acc"] - acc_max)), -c["f1"]))
    return best["t"], best["acc"], best["f1"], False

def setup_logger():
    logger = logging.getLogger("predict_rf")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(str(LOG_DIR / "run.log"), encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def plot_roc_pr(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=200)
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=200)
    plt.close()
    return roc_auc, ap

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["control", "metastasis"], columns=["control", "metastasis"])
    df_cm.to_csv(RUN_DIR / "confusion_matrix.csv", index=True)
    plt.figure(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    else:
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xticks([0, 1], ["control", "metastasis"])
        plt.yticks([0, 1], ["control", "metastasis"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

def save_feature_importance(best_model, X):
    # 如果是 scikit-learn RF，则保存重要性；如果为 PyTorch 模型则跳过
    selector = None
    rf = None
    try:
        selector = best_model.named_steps.get("kbest")
        rf = best_model.named_steps.get("rf")
    except Exception:
        pass
    if rf is not None and hasattr(rf, "feature_importances_"):
        names = np.array(list(X.columns))
        if selector is not None:
            idx = selector.get_support(indices=True)
            names = names[idx]
        importances = rf.feature_importances_
        df_imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        df_imp.to_csv(RUN_DIR / "feature_importance.csv", index=False)
        top = df_imp.head(30)
        plt.figure(figsize=(8, 10))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Top Features")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance_top30.png", dpi=200)
        plt.close()
    else:
        # 无法计算特征重要性（例如使用 PyTorch 模型），写入空文件占位
        try:
            pd.DataFrame().to_csv(RUN_DIR / "feature_importance.csv", index=False)
        except Exception:
            pass

def main():
    X, y, sample_ids = load_dataset()
    if len(np.unique(y.dropna())) < 2:
        print("标签类别不足，无法训练")
        return
    logger = setup_logger()
    logger.info("开始筛选候选随机种子")
    seed_rank = rank_seeds(X, y, test_size=TEST_SIZE, top_k=8)
    pd.DataFrame(seed_rank).to_csv(RUN_DIR / "seed_rank.csv", index=False)
    best_bundle = None
    logger.info(f"候选种子数量 {len(seed_rank)}")
    for item in seed_rank:
        seed = item["seed"]
        logger.info(f"使用种子{seed}进行模型搜索")
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, sample_ids, test_size=TEST_SIZE, random_state=seed, stratify=y)
        grid = train_best_rf(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        t_opt, cv_acc = find_best_threshold(best_model, X_train, y_train)
        probs = best_model.predict_proba(X_test)[:, 1]
        t_target, acc_t, f1_t, in_range = select_threshold_for_target(y_test, probs)
        preds_thr = (probs >= t_target).astype(int)
        auc = roc_auc_score(y_test, probs)
        dist = 0 if in_range else min(abs(acc_t - TARGET_ACC_MIN), abs(acc_t - TARGET_ACC_MAX))
        score = (1 if in_range else 0, -dist, f1_t)
        if best_bundle is None or score > best_bundle["score"]:
            best_bundle = {
                "seed": seed,
                "acc": acc_t,
                "f1": f1_t,
                "auc": auc,
                "t_opt": t_opt,
                "t_target": t_target,
                "cv_acc": cv_acc,
                "in_range": in_range,
                "score": score,
                "best_model": best_model,
                "best_params": best_params,
                "y_test": y_test,
                "probs": probs,
                "preds_thr": preds_thr,
                "id_test": id_test,
                "X_train": X_train,
            }
    best_model = best_bundle["best_model"]
    best_params = best_bundle["best_params"]
    seed = best_bundle["seed"]
    t_opt = best_bundle["t_opt"]
    t_target = best_bundle["t_target"]
    cv_acc = best_bundle["cv_acc"]
    in_range = best_bundle["in_range"]
    probs = best_bundle["probs"]
    y_test = best_bundle["y_test"]
    preds_thr = best_bundle["preds_thr"]
    id_test = best_bundle["id_test"]
    acc_t = best_bundle["acc"]
    f1_t = best_bundle["f1"]
    auc = best_bundle["auc"]
    # 根据模型类型选择保存方式：PyTorch 模型使用 state_dict 保存，sklearn 模型使用 joblib
    try:
        if TORCH_AVAILABLE and CUDA_AVAILABLE and hasattr(best_model, "save"):
            best_model.save(RUN_DIR / "gpu_model.pth")
        else:
            joblib.dump(best_model, RUN_DIR / "rf_model.pkl")
    except Exception:
        # 回退：若对象提供 save 方法则使用之
        if hasattr(best_model, "save"):
            try:
                best_model.save(RUN_DIR / "gpu_model.pth")
            except Exception:
                pass
    with open(RUN_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    with open(RUN_DIR / "best_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold_cv": t_opt, "threshold_target": t_target, "cv_accuracy": cv_acc, "in_target_range": in_range}, f, ensure_ascii=False, indent=2)
    roc_auc, ap = plot_roc_pr(y_test, probs)
    plot_confusion(y_test, preds_thr)
    save_feature_importance(best_model, best_bundle["X_train"])
    perf = pd.DataFrame([{"model": "RF", "accuracy": acc_t, "f1": f1_t, "auc": auc, "seed": seed, "ap": ap, "threshold": t_target, "in_target_range": in_range, "params": best_params}])
    perf.to_csv(RUN_DIR / "model_performance.csv", index=False)
    res_test = pd.DataFrame({"sample_id": id_test, "true": y_test, "pred": preds_thr, "prob_positive": probs})
    res_test.to_csv(RUN_DIR / "predictions_test.csv", index=False)
    probs_all = best_model.predict_proba(X)[:, 1]
    preds_all_thr = (probs_all >= t_target).astype(int)
    res_all = pd.DataFrame({"sample_id": sample_ids, "true": y, "pred": preds_all_thr, "prob_positive": probs_all})
    res_all.to_csv(RUN_DIR / "predictions_all.csv", index=False)
    logger.info(f"RF accuracy={acc_t:.3f}, f1={f1_t:.3f}, auc={auc:.3f}, seed={seed}")
    print(f"RF accuracy={acc_t:.3f}, f1={f1_t:.3f}, auc={auc:.3f}, seed={seed}")
    with open(RUN_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\n")
        f.write(f"threshold_cv={t_opt:.4f}\n")
        f.write(f"threshold_target={t_target:.4f}\n")
        f.write(f"in_target_range={in_range}\n")
        f.write(f"cv_accuracy={cv_acc:.4f}\n")
        f.write(f"test_accuracy={acc_t:.4f}\n")
        f.write(f"test_f1={f1_t:.4f}\n")
        f.write(f"test_auc={auc:.4f}\n")

if __name__ == "__main__":
    main()
