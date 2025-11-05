import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
def lasso_feature_selection(X, y, threshold=0.01, n_features=30):
    """使用LASSO回归进行特征选择，优化版本"""
    print("Performing LASSO feature selection...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用LassoCV进行交叉验证选择最优参数，增加交叉验证折数和迭代次数
    lasso_cv = LassoCV(cv=10, random_state=42, max_iter=20000, n_jobs=-1)
    lasso_cv.fit(X_scaled, y)
    
    # 获取特征系数
    coefficients = lasso_cv.coef_
    
    # 选择非零系数的特征
    selected_features_mask = np.abs(coefficients) > threshold
    selected_features = X.columns[selected_features_mask]
    
    # 优化特征选择策略：确保选择足够的特征，根据系数绝对值排序
    if len(selected_features) < n_features:
        # 如果选择的特征不足，根据系数绝对值选择最重要的n_features个
        coef_abs = np.abs(coefficients)
        top_indices = np.argsort(coef_abs)[::-1][:n_features]
        selected_features = X.columns[top_indices]
        print(f"Adjusted to {len(selected_features)} top features based on coefficient magnitude")
    elif len(selected_features) > n_features * 2:
        # 如果选择的特征太多，选择系数绝对值最大的特征
        coef_abs = np.abs(coefficients[selected_features_mask])
        top_indices = np.argsort(coef_abs)[::-1][:n_features]
        selected_features = selected_features[top_indices]
        print(f"Reduced to {len(selected_features)} top features based on coefficient magnitude")
    
    print(f"LASSO selected {len(selected_features)} features")
    return X[selected_features], selected_features, coefficients

# 模拟WGCNA模块特征提取（基于相关性分析）
def wgcna_module_features(X, n_modules=8):
    """改进的WGCNA模块分析，基于特征相关性聚类和模块内特征综合"""
    print("Extracting WGCNA-like module features...")
    
    # 限制特征数量以避免计算过大的相关性矩阵，增加特征数量以捕获更多信息
    max_features = min(1500, X.shape[1])
    if X.shape[1] > max_features:
        # 先选择方差最大的特征
        top_var_genes = X.var().nlargest(max_features).index
        X_reduced = X[top_var_genes]
        print(f"Reducing feature set from {X.shape[1]} to {X_reduced.shape[1]} for correlation analysis")
    else:
        X_reduced = X
    
    # 计算特征相关性矩阵
    corr_matrix = X_reduced.corr()
    
    # 为每个模块创建一个特征（改进：取模块内多个特征的加权平均）
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
        
        # 改进：找到与中心基因高度相关的特征，计算模块内特征的加权平均
        # 找出与中心基因相关系数大于0.6的特征
        related_genes = corr_matrix[module_gene][abs(corr_matrix[module_gene]) > 0.6].index.tolist()
        
        if len(related_genes) > 1:
            # 计算加权平均（相关性作为权重）
            weights = abs(corr_matrix[module_gene][related_genes])
            weights = weights / weights.sum()  # 归一化权重
            
            # 创建加权平均特征
            module_avg = np.zeros(len(X))
            for j, gene in enumerate(related_genes):
                module_avg += X[gene].values * weights.iloc[j]
            
            module_features[f"module_{i+1}"] = module_avg
        else:
            # 如果没有足够的相关特征，只使用中心基因
            module_features[f"module_{i+1}"] = X[module_gene]
    
    print(f"Extracted {module_features.shape[1]} WGCNA-like module features")
    return module_features

# 差异表达基因分析（t-test）
def differential_expression_analysis(X, y, top_n=60):
    """改进的差异表达分析，使用更稳健的统计方法和效应量"""
    print("Performing differential expression analysis...")
    
    # 将样本分为两组
    group0 = X[y == 0]
    group1 = X[y == 1]
    
    # 计算每个特征在两组间的差异统计量
    diff_results = []
    for col in X.columns:
        # 使用t-test
        try:
            # 使用Welch's t-test（不等方差）
            t_stat, t_pval = stats.ttest_ind(group0[col], group1[col], equal_var=False)
            
            # 计算效应量（Cohen's d）
            mean0 = group0[col].mean()
            mean1 = group1[col].mean()
            mean_diff = mean1 - mean0
            
            # 计算合并标准差
            n0, n1 = len(group0), len(group1)
            var0, var1 = group0[col].var(), group1[col].var()
            pooled_std = np.sqrt(((n0-1)*var0 + (n1-1)*var1) / (n0 + n1 - 2)) if n0 + n1 > 2 else 1
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # 综合分数：结合t统计量和效应量
            combined_score = abs(t_stat) * min(abs(cohens_d), 3)  # 限制效应量的影响范围
            
            diff_results.append((col, abs(t_stat), t_pval, cohens_d, combined_score))
        except:
            # 处理计算错误
            diff_results.append((col, 0, 1.0, 0, 0))
    
    # 转换为DataFrame并按综合分数排序
    results_df = pd.DataFrame(diff_results, 
                             columns=['feature', 'abs_t_stat', 'p_value', 'cohens_d', 'combined_score'])
    results_df = results_df.sort_values('combined_score', ascending=False)
    
    # 选择top差异表达基因
    top_de_genes = results_df['feature'].head(top_n).tolist()
    
    print(f"Selected {len(top_de_genes)} top differentially expressed genes")
    return X[top_de_genes], top_de_genes

# 整合特征选择方法
def integrated_feature_selection(X, y):
    """整合多种特征选择方法，增加特征重要性权重和特征过滤"""
    # 1. LASSO特征选择
    X_lasso, lasso_features, lasso_coefs = lasso_feature_selection(X, y, n_features=30)
    
    # 2. 差异表达基因分析
    X_de, de_features = differential_expression_analysis(X, y, top_n=60)
    
    # 3. WGCNA模块特征
    X_wgcna = wgcna_module_features(X, n_modules=8)
    
    # 4. 特征重要性评分
    feature_scores = {}
    
    # 为LASSO特征分配权重（基于系数绝对值）
    lasso_coef_dict = dict(zip(lasso_features, np.abs(lasso_coefs[np.isin(X.columns, lasso_features)])))
    lasso_max_coef = max(lasso_coef_dict.values()) if lasso_coef_dict else 1
    for feature, coef in lasso_coef_dict.items():
        feature_scores[feature] = feature_scores.get(feature, 0) + 3 * (coef / lasso_max_coef)  # LASSO权重最高
    
    # 为差异表达特征分配权重
    for feature in de_features[:30]:  # 前30个DE特征权重更高
        feature_scores[feature] = feature_scores.get(feature, 0) + 2
    for feature in de_features[30:]:  # 剩余DE特征权重较低
        feature_scores[feature] = feature_scores.get(feature, 0) + 1
    
    # 找出被多种方法选中的特征（可能更重要）
    all_features = list(set(lasso_features) | set(de_features))
    overlap_features = [f for f in all_features if f in lasso_features and f in de_features]
    
    # 为重叠特征增加额外权重
    for feature in overlap_features:
        feature_scores[feature] = feature_scores.get(feature, 0) + 1  # 重叠特征额外加分
    
    # 根据重要性分数选择特征
    sorted_features = sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)
    
    # 限制特征数量以避免过拟合
    max_features = min(100, len(sorted_features))
    final_features = sorted_features[:max_features]
    
    # 创建最终特征集
    X_selected = X[final_features].copy()
    
    # 添加WGCNA模块特征
    for col in X_wgcna.columns:
        X_selected[col] = X_wgcna[col]
    
    # 打印特征选择统计信息
    print(f"Final integrated features count: {X_selected.shape[1]}")
    if overlap_features:
        print(f"Found {len(overlap_features)} features selected by multiple methods (potentially more reliable)")
    
    # 返回特征重要性分数字典，用于后续分析
    return X_selected, feature_scores

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
    X_train_selected, feature_scores = integrated_feature_selection(X_train, y_train)
    
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
    
    # 创建集成分类器，结合多种算法
    print("Creating ensemble model with multiple algorithms...")
    
    # 定义基础分类器
    classifiers = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'svc': SVC(probability=True, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # 为每个分类器进行参数优化
    best_classifiers = {}
    
    # 随机森林参数网格
    rf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 15, 25],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_search = RandomizedSearchCV(
        classifiers['rf'],
        param_distributions=rf_param_dist,
        n_iter=15,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )
    rf_search.fit(X_train_selected, y_train)
    best_classifiers['rf'] = rf_search.best_estimator_
    print(f"Best Random Forest params: {rf_search.best_params_}")
    print(f"Random Forest CV score: {rf_search.best_score_:.4f}")
    
    # 梯度提升参数网格
    gb_param_dist = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    gb_search = RandomizedSearchCV(
        classifiers['gb'],
        param_distributions=gb_param_dist,
        n_iter=15,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )
    gb_search.fit(X_train_selected, y_train)
    best_classifiers['gb'] = gb_search.best_estimator_
    print(f"Best Gradient Boosting params: {gb_search.best_params_}")
    print(f"Gradient Boosting CV score: {gb_search.best_score_:.4f}")
    
    # 创建软投票集成分类器
    search = VotingClassifier(
        estimators=[
            ('rf', best_classifiers['rf']),
            ('gb', best_classifiers['gb']),
            ('svc', classifiers['svc']),  # SVC使用默认参数
            ('lr', classifiers['lr'])     # 逻辑回归使用默认参数
        ],
        voting='soft',  # 使用概率加权
        weights=[1, 1, 1, 1]
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
    
    # 可视化结果和特征重要性分析
    print("\n生成可视化结果和特征重要性分析...")
    visualize_results(X_test_selected, y_test, preds, probs, search, classifiers, best_classifiers, feature_scores, out_dir)

# 可视化和特征重要性分析函数
def visualize_results(X_test, y_test, preds, probs, model, classifiers, best_classifiers, feature_scores, out_dir):
    """生成可视化结果和特征重要性分析"""
    # 创建可视化子目录
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. ROC曲线
    if probs is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        roc_path = os.path.join(viz_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存: {roc_path}")
    
    # 2. 精确率-召回率曲线
    if probs is not None:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR曲线 (面积 = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="best")
        pr_path = os.path.join(viz_dir, 'precision_recall_curve.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"精确率-召回率曲线已保存: {pr_path}")
    
    # 2. 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['无转移', '有转移'], yticklabels=['无转移', '有转移'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵可视化已保存: {cm_path}")
    
    # 3. 特征重要性排序图（基于特征选择分数）
    # 过滤掉模块特征，只显示原始特征
    original_feature_scores = {k: v for k, v in feature_scores.items() if not k.startswith('module_')}
    sorted_features = sorted(original_feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 显示前20个最重要的特征
    top_features = sorted_features[:20]
    feature_names = [item[0] for item in top_features]
    importance_values = [item[1] for item in top_features]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), importance_values, align='center')
    plt.yticks(range(len(top_features)), feature_names)
    plt.xlabel('特征重要性分数')
    plt.title('前20个最重要的特征')
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    features_path = os.path.join(viz_dir, 'top_features.png')
    plt.savefig(features_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性排序图已保存: {features_path}")
    
    # 4. 随机森林特征重要性（如果随机森林是集成模型的一部分）
    if 'rf' in best_classifiers:
        rf_importances = best_classifiers['rf'].feature_importances_
        # 只考虑原始特征（非模块特征）
        original_indices = [i for i, col in enumerate(X_test.columns) if not col.startswith('module_')]
        rf_importances_original = rf_importances[original_indices]
        original_cols = [X_test.columns[i] for i in original_indices]
        
        # 排序并取前15个
        rf_sorted_idx = np.argsort(rf_importances_original)[::-1][:15]
        rf_top_features = [original_cols[i] for i in rf_sorted_idx]
        rf_top_importances = rf_importances_original[rf_sorted_idx]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(rf_top_features)), rf_top_importances, align='center')
        plt.yticks(range(len(rf_top_features)), rf_top_features)
        plt.xlabel('随机森林特征重要性')
        plt.title('随机森林模型中的前15个重要特征')
        plt.gca().invert_yaxis()
        rf_path = os.path.join(viz_dir, 'rf_feature_importance.png')
        plt.savefig(rf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"随机森林特征重要性图已保存: {rf_path}")
    
    # 5. 预测概率分布图
    if probs is not None:
        plt.figure(figsize=(10, 6))
        # 分离正负样本的预测概率
        y_test_np = np.array(y_test)
        probs_pos = probs[y_test_np == 1]
        probs_neg = probs[y_test_np == 0]
        
        # 绘制直方图
        plt.hist(probs_neg, bins=20, alpha=0.5, label='无转移', color='blue')
        plt.hist(probs_pos, bins=20, alpha=0.5, label='有转移', color='red')
        plt.xlabel('预测概率 (有转移)')
        plt.ylabel('样本数量')
        plt.title('预测概率分布')
        plt.legend()
        prob_dist_path = os.path.join(viz_dir, 'prediction_probability_distribution.png')
        plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测概率分布图已保存: {prob_dist_path}")
    
    # 6. 各分类器性能比较
    classifiers_performance = []
    
    # 先评估集成模型
    ensemble_acc = accuracy_score(y_test, preds)
    classifiers_performance.append(('集成模型', ensemble_acc))
    
    # 然后评估各个基础分类器
    for name, clf in best_classifiers.items():
        clf_preds = clf.predict(X_test)
        clf_acc = accuracy_score(y_test, clf_preds)
        classifiers_performance.append((name.upper(), clf_acc))
    
    # 绘制性能比较图
    models = [item[0] for item in classifiers_performance]
    accuracies = [item[1] for item in classifiers_performance]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['green'] + ['blue'] * (len(models) - 1))
    plt.ylim(0, 1)
    plt.ylabel('准确率')
    plt.title('各分类器性能比较')
    
    # 在柱状图上添加准确率数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    performance_path = os.path.join(viz_dir, 'classifiers_performance.png')
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分类器性能比较图已保存: {performance_path}")
    
    # 保存特征重要性到CSV文件
    features_df = pd.DataFrame(sorted_features, columns=['特征名', '重要性分数'])
    features_df = features_df.head(50)  # 只保存前50个最重要的特征
    features_csv_path = os.path.join(viz_dir, 'feature_importance.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"特征重要性数据已保存: {features_csv_path}")
    
    # 打印最重要的特征列表
    print("\n影响较大的前15个特征:")
    for i, (feature, score) in enumerate(sorted_features[:15]):
        print(f"{i+1}. {feature}: {score:.4f}")

if __name__ == "__main__":
    main()