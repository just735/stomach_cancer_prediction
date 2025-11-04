import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据加载函数
def load_data():
    """加载合并后的数据"""
    data_path = os.path.join(BASE_DIR, "data", "data_merged.csv")
    try:
        # 读取预处理后的数据文件
        data = pd.read_csv(data_path, index_col=0)
        print(f"Loaded merged data: {os.path.basename(data_path)} -> samples={data.shape[0]}, columns={data.shape[1]}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# 基于LASSO的特征选择（模仿R脚本中的lasso.R）
def lasso_feature_selection(X, y, threshold=0.01):
    """使用LASSO回归进行特征选择"""
    print("Performing LASSO feature selection...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用LassoCV进行交叉验证选择最优参数
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y)
    
    # 获取特征系数
    coefficients = lasso_cv.coef_
    
    # 选择非零系数的特征
    selected_features_mask = np.abs(coefficients) > threshold
    selected_features = X.columns[selected_features_mask]
    
    print(f"LASSO selected {len(selected_features)} features")
    return X[selected_features], selected_features, coefficients

# 模拟WGCNA模块特征提取（基于相关性分析）
def wgcna_module_features(X, n_modules=5):
    """模拟WGCNA模块分析，基于特征相关性聚类"""
    print("Extracting WGCNA-like module features...")
    
    # 限制特征数量以避免计算过大的相关性矩阵
    if X.shape[1] > 1000:
        # 先选择方差最大的1000个特征
        top_var_genes = X.var().nlargest(1000).index
        X_reduced = X[top_var_genes]
        print(f"Reducing feature set from {X.shape[1]} to {X_reduced.shape[1]} for correlation analysis")
    else:
        X_reduced = X
    
    # 计算特征相关性矩阵
    corr_matrix = X_reduced.corr()
    
    # 为每个模块创建一个特征（取模块内特征的平均值）
    module_features = pd.DataFrame(index=X.index)
    selected_genes = []
    
    # 选择相关性最高的n_modules组特征作为模块代表
    for i in range(n_modules):
        # 找出当前最相关的特征对
        if i == 0:
            # 第一个模块：选择方差最大的特征
            module_gene = X_reduced.var().idxmax()
        else:
            # 后续模块：选择与已有模块基因相关性最低的特征
            remaining_genes = [g for g in X_reduced.columns if g not in selected_genes]
            if not remaining_genes:
                break
            
            # 计算每个剩余特征与已选基因的平均相关性
            avg_corrs = []
            for gene in remaining_genes:
                # 计算该基因与所有已选基因的平均相关性
                corrs = [np.abs(corr_matrix.loc[gene, sel_gene]) for sel_gene in selected_genes]
                avg_corr = np.mean(corrs) if corrs else 0
                avg_corrs.append(avg_corr)
            
            # 选择与已有模块相关性最低的基因
            module_gene = remaining_genes[np.argmin(avg_corrs)]
        
        # 将模块特征添加到结果中
        selected_genes.append(module_gene)
        module_features[f"module_{i+1}"] = X[module_gene]
    
    print(f"Extracted {module_features.shape[1]} WGCNA-like module features")
    return module_features

# 差异表达基因分析（t-test）
def differential_expression_analysis(X, y):
    """执行简单的差异表达分析，选择表达差异最显著的特征"""
    print("Performing differential expression analysis...")
    
    # 将样本分为两组
    group0 = X[y == 0]
    group1 = X[y == 1]
    
    # 计算每个特征在两组间的平均差异和t统计量
    diff_scores = []
    for col in X.columns:
        mean0 = group0[col].mean()
        mean1 = group1[col].mean()
        std0 = group0[col].std() if len(group0) > 1 else 1
        std1 = group1[col].std() if len(group1) > 1 else 1
        
        # 计算t统计量（简化版）
        mean_diff = mean1 - mean0
        pooled_std = np.sqrt(std0**2/len(group0) + std1**2/len(group1))
        t_stat = mean_diff / pooled_std if pooled_std > 0 else 0
        
        diff_scores.append((col, abs(t_stat)))
    
    # 按t统计量排序
    diff_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 选择top差异表达基因
    top_de_genes = [gene for gene, score in diff_scores[:50]]  # 选择top50
    
    print(f"Selected {len(top_de_genes)} top differentially expressed genes")
    return X[top_de_genes], top_de_genes

# 整合特征选择方法
def integrated_feature_selection(X, y):
    """整合多种特征选择方法"""
    # 1. LASSO特征选择
    X_lasso, lasso_features, _ = lasso_feature_selection(X, y)
    
    # 2. 差异表达基因分析
    X_de, de_features = differential_expression_analysis(X, y)
    
    # 3. WGCNA模块特征
    X_wgcna = wgcna_module_features(X)
    
    # 4. 合并所有特征
    selected_features = list(set(lasso_features) | set(de_features))
    X_selected = X[selected_features].copy()
    
    # 添加WGCNA模块特征
    for col in X_wgcna.columns:
        X_selected[col] = X_wgcna[col]
    
    print(f"Final integrated features count: {X_selected.shape[1]}")
    return X_selected

# 主函数
def main():
    # 创建输出目录
    out_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    data = load_data()
    
    # 提取特征和标签
    # 假设标签列是'characteristics_ch1.15'（从WGCNA1.R中推断）
    if 'characteristics_ch1.15' in data.columns:
        y = data['characteristics_ch1.15']
        # 转换为0/1标签（如果需要）
        if y.dtype == 'object':
            y = y.map({'no': 0, 'yes': 1})
        X = data.drop('characteristics_ch1.15', axis=1)
    else:
        # 如果找不到明确的标签列，假设最后一列是标签
        print("Warning: 'characteristics_ch1.15' not found. Assuming last column is label.")
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 应用整合特征选择
    X_train_selected = integrated_feature_selection(X_train, y_train)
    
    # 对测试集应用相同的特征选择
    common_features = [col for col in X_train_selected.columns if col in X_test.columns]
    wgcna_features = [col for col in X_train_selected.columns if col.startswith('module_')]
    
    X_test_selected = X_test[common_features].copy()
    
    # 对于WGCNA模块特征，从原始X_test中计算
    if wgcna_features:
        X_test_wgcna = wgcna_module_features(X_test)
        for col in wgcna_features:
            if col in X_test_wgcna.columns:
                X_test_selected[col] = X_test_wgcna[col]
    
    # 确保训练集和测试集特征一致
    missing_cols = set(X_train_selected.columns) - set(X_test_selected.columns)
    if missing_cols:
        print(f"Warning: {len(missing_cols)} features in training not in test. Adding zeros.")
        for col in missing_cols:
            X_test_selected[col] = 0
    
    # 保持特征顺序一致
    X_test_selected = X_test_selected[X_train_selected.columns]
    
    # 设置随机森林参数网格
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 随机搜索最佳参数
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    print("Training model with integrated features...")
    search.fit(X_train_selected, y_train)
    
    # 预测
    preds = search.predict(X_test_selected)
    probs = None
    if hasattr(search, "predict_proba"):
        probs_ = search.predict_proba(X_test_selected)
        probs = probs_[:, 1] if probs_.shape[1] == 2 else None
    
    # 对所有样本进行预测
    # 分离原始特征和模块特征
    original_features = [col for col in X_train_selected.columns if not col.startswith('module_')]
    module_feature_names = [col for col in X_train_selected.columns if col.startswith('module_')]
    
    # 提取原始特征
    X_all_original = X[original_features]
    
    # 为所有样本计算模块特征
    X_all_wgcna = wgcna_module_features(X)
    
    # 合并原始特征和模块特征
    X_all_selected = X_all_original.copy()
    for col in module_feature_names:
        if col in X_all_wgcna.columns:
            X_all_selected[col] = X_all_wgcna[col]
    
    # 确保特征顺序一致
    X_all_selected = X_all_selected[X_train_selected.columns]
    
    # 进行预测
    preds_all = search.predict(X_all_selected)
    probs_all = None
    if hasattr(search, "predict_proba"):
        probs_all_ = search.predict_proba(X_all_selected)
        probs_all = probs_all_[:, 1] if probs_all_.shape[1] == 2 else None
    
    # 保存预测结果
    # 构造测试集输出表
    df_out = pd.DataFrame({"true": y_test, "pred": preds}, index=X_test.index)
    if probs is not None:
        df_out["prob_positive"] = probs
    csv_path = os.path.join(out_dir, "predictions_integrated.csv")
    df_out.to_csv(csv_path, index=True)
    print(f"预测已保存: {csv_path}")
    
    # 保存全部样本预测
    out_all = pd.DataFrame({"true": y, "pred": preds_all}, index=X.index)
    if probs_all is not None:
        out_all["prob_positive"] = probs_all
    all_csv_path = os.path.join(out_dir, "predictions_all_integrated.csv")
    out_all.to_csv(all_csv_path, index=True)
    print(f"已保存全部样本预测到: {all_csv_path}")
    
    # 评估指标
    acc = accuracy_score(y_test, preds)
    print(f"准确率: {acc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, preds)
    print("混淆矩阵:")
    print(cm)
    
    # 分类报告
    report = classification_report(y_test, preds)
    print("\n分类报告:")
    print(report)
    
    # 数据集信息
    print(f"\n原始样本数: {X.shape[0]}")
    print(f"X 总行数: {X.shape[0]}")
    print(f"X_train 行数: {X_train.shape[0]}")
    print(f"X_test  行数: {X_test.shape[0]}")
    print(f"最终特征数: {X_train_selected.shape[1]}")

if __name__ == "__main__":
    main()