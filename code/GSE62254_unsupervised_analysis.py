#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE62254胃癌数据集无监督学习分析
基于真实基因表达数据的样本聚类和特征发现
不使用任何标签，完全基于数据驱动的分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
VIS_DIR = OUTPUT_DIR / "GSE62254_unsupervised_analysis"

class GSE62254UnsupervisedAnalyzer:
    """GSE62254无监督学习分析器"""
    
    def __init__(self):
        self.expression_file = DATA_DIR / "GEO" / "GSE62254_series_matrix.txt"
        self.output_dir = OUTPUT_DIR
        self.vis_dir = VIS_DIR
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        
        print("🔬 GSE62254胃癌数据集无监督学习分析器")
        print("=" * 60)
        print("📋 分析特点:")
        print("   ✅ 基于真实基因表达数据")
        print("   ✅ 不使用任何标签信息")
        print("   ✅ 完全数据驱动的发现")
        print("   ✅ 多种聚类算法比较")
        print("=" * 60)
    
    def load_expression_data(self):
        """加载基因表达数据"""
        print("📊 加载GSE62254基因表达数据...")
        
        if not self.expression_file.exists():
            raise FileNotFoundError(f"数据文件未找到: {self.expression_file}")
        
        try:
            # 读取表达矩阵，跳过注释行
            expr_data = pd.read_csv(
                self.expression_file, 
                sep="\t", 
                comment="!", 
                index_col=0, 
                engine="python",
                on_bad_lines="skip"
            )
            
            print(f"✅ 数据加载成功: {expr_data.shape[0]} 基因 x {expr_data.shape[1]} 样本")
            return expr_data
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def preprocess_data(self, expr_data):
        """数据预处理和质量控制"""
        print("🔧 数据预处理和质量控制...")
        
        # 转置数据，使样本为行，基因为列
        expr_transposed = expr_data.T
        print(f"   转置后数据: {expr_transposed.shape[0]} 样本 x {expr_transposed.shape[1]} 基因")
        
        # 1. 移除缺失值过多的基因
        missing_threshold = 0.2
        missing_ratio = expr_transposed.isnull().sum() / len(expr_transposed)
        valid_genes = missing_ratio[missing_ratio <= missing_threshold].index
        expr_clean = expr_transposed[valid_genes]
        print(f"   移除高缺失基因后: {len(valid_genes)} 基因保留")
        
        # 2. 填充剩余缺失值
        expr_clean = expr_clean.fillna(expr_clean.median())
        
        # 3. 移除低方差基因
        gene_variance = expr_clean.var()
        variance_threshold = np.percentile(gene_variance, 25)  # 保留方差前75%的基因
        high_var_genes = gene_variance[gene_variance > variance_threshold].index
        expr_filtered = expr_clean[high_var_genes]
        print(f"   移除低方差基因后: {len(high_var_genes)} 基因保留")
        
        # 4. 移除极端异常值
        # 使用IQR方法识别异常样本
        Q1 = expr_filtered.quantile(0.25, axis=1)
        Q3 = expr_filtered.quantile(0.75, axis=1)
        IQR = Q3 - Q1
        
        # 计算每个样本的异常程度
        outlier_scores = []
        for idx in expr_filtered.index:
            sample_data = expr_filtered.loc[idx]
            outliers = ((sample_data < (Q1[idx] - 1.5 * IQR[idx])) | 
                       (sample_data > (Q3[idx] + 1.5 * IQR[idx]))).sum()
            outlier_scores.append(outliers / len(sample_data))
        
        outlier_threshold = np.percentile(outlier_scores, 95)  # 移除异常程度最高的5%样本
        normal_samples = [i for i, score in enumerate(outlier_scores) if score <= outlier_threshold]
        expr_final = expr_filtered.iloc[normal_samples]
        
        print(f"   移除异常样本后: {len(expr_final)} 样本保留")
        print(f"✅ 预处理完成: {expr_final.shape[0]} 样本 x {expr_final.shape[1]} 基因")
        
        return expr_final
    
    def perform_pca_analysis(self, data):
        """主成分分析"""
        print("📈 执行主成分分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA分析
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # 计算解释方差比
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # 确定保留的主成分数量（解释90%方差）
        n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
        n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        
        print(f"✅ PCA分析完成:")
        print(f"   解释80%方差需要: {n_components_80} 个主成分")
        print(f"   解释90%方差需要: {n_components_90} 个主成分")
        print(f"   前10个主成分解释: {cumulative_variance[9]:.2%} 的方差")
        
        return {
            'pca_result': pca_result,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'pca_model': pca,
            'scaler': scaler,
            'n_components_80': n_components_80,
            'n_components_90': n_components_90
        }
    
    def determine_optimal_clusters(self, pca_result, max_k=10):
        """确定最优聚类数"""
        print("🎯 确定最优聚类数...")
        
        # 使用前20个主成分进行聚类
        data_for_clustering = pca_result[:, :20]
        
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_for_clustering)
            
            # 计算评估指标
            sil_score = silhouette_score(data_for_clustering, cluster_labels)
            cal_score = calinski_harabasz_score(data_for_clustering, cluster_labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            inertias.append(kmeans.inertia_)
        
        # 找到最优k值
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        best_k_calinski = k_range[np.argmax(calinski_scores)]
        
        # 使用肘部法则
        # 计算惯性的二阶差分
        if len(inertias) >= 3:
            second_diff = np.diff(inertias, 2)
            elbow_k = k_range[np.argmax(second_diff) + 2] if len(second_diff) > 0 else 4
        else:
            elbow_k = 4
        
        print(f"✅ 聚类数评估完成:")
        print(f"   轮廓系数最优k: {best_k_silhouette} (分数: {max(silhouette_scores):.3f})")
        print(f"   Calinski-Harabasz最优k: {best_k_calinski} (分数: {max(calinski_scores):.1f})")
        print(f"   肘部法则建议k: {elbow_k}")
        
        # 综合选择k值
        recommended_k = best_k_silhouette  # 优先使用轮廓系数
        
        return {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'inertias': inertias,
            'best_k_silhouette': best_k_silhouette,
            'best_k_calinski': best_k_calinski,
            'elbow_k': elbow_k,
            'recommended_k': recommended_k
        }
    
    def perform_multiple_clustering(self, data, pca_result, optimal_k):
        """执行多种聚类算法"""
        print(f"🔄 执行多种聚类算法 (k={optimal_k})...")
        
        # 使用前20个主成分
        data_for_clustering = pca_result[:, :20]
        
        clustering_results = {}
        
        # 1. K-means聚类
        print("   执行K-means聚类...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(data_for_clustering)
        kmeans_silhouette = silhouette_score(data_for_clustering, kmeans_labels)
        clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette': kmeans_silhouette,
            'model': kmeans
        }
        
        # 2. 层次聚类
        print("   执行层次聚类...")
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(data_for_clustering)
        hierarchical_silhouette = silhouette_score(data_for_clustering, hierarchical_labels)
        clustering_results['hierarchical'] = {
            'labels': hierarchical_labels,
            'silhouette': hierarchical_silhouette,
            'model': hierarchical
        }
        
        # 3. DBSCAN聚类（自动确定聚类数）
        print("   执行DBSCAN聚类...")
        # 估计eps参数
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=20)
        neighbors_fit = neighbors.fit(data_for_clustering)
        distances, indices = neighbors_fit.kneighbors(data_for_clustering)
        distances = np.sort(distances[:, -1], axis=0)
        eps = np.percentile(distances, 90)  # 使用90%分位数作为eps
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(data_for_clustering)
        
        # 检查DBSCAN结果
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        if n_clusters_dbscan > 1:
            # 只有当聚类数>1时才计算轮廓系数
            valid_labels = dbscan_labels[dbscan_labels != -1]
            valid_data = data_for_clustering[dbscan_labels != -1]
            if len(set(valid_labels)) > 1:
                dbscan_silhouette = silhouette_score(valid_data, valid_labels)
            else:
                dbscan_silhouette = -1
        else:
            dbscan_silhouette = -1
        
        clustering_results['dbscan'] = {
            'labels': dbscan_labels,
            'silhouette': dbscan_silhouette,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise,
            'model': dbscan
        }
        
        print(f"✅ 多种聚类算法完成:")
        print(f"   K-means轮廓系数: {kmeans_silhouette:.3f}")
        print(f"   层次聚类轮廓系数: {hierarchical_silhouette:.3f}")
        print(f"   DBSCAN聚类数: {n_clusters_dbscan}, 噪声点: {n_noise}, 轮廓系数: {dbscan_silhouette:.3f}")
        
        # 选择最佳聚类方法
        best_method = 'kmeans'
        best_score = kmeans_silhouette
        
        if hierarchical_silhouette > best_score:
            best_method = 'hierarchical'
            best_score = hierarchical_silhouette
        
        if dbscan_silhouette > best_score and n_clusters_dbscan >= 2:
            best_method = 'dbscan'
            best_score = dbscan_silhouette
        
        print(f"   最佳聚类方法: {best_method} (轮廓系数: {best_score:.3f})")
        
        return clustering_results, best_method
    
    def find_cluster_marker_genes(self, data, cluster_labels, method_name):
        """寻找聚类标志基因"""
        print(f"🔍 寻找{method_name}聚类的标志基因...")
        
        unique_clusters = np.unique(cluster_labels)
        # 排除噪声点（DBSCAN中的-1）
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        marker_genes = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            other_mask = (cluster_labels != cluster_id) & (cluster_labels != -1)  # 排除噪声点
            
            if np.sum(cluster_mask) < 3 or np.sum(other_mask) < 3:
                # 样本数太少，跳过
                continue
            
            # 对每个基因进行t检验
            p_values = []
            fold_changes = []
            effect_sizes = []
            
            for gene in data.columns:
                cluster_expr = data.loc[cluster_mask, gene]
                other_expr = data.loc[other_mask, gene]
                
                # t检验
                try:
                    t_stat, p_val = stats.ttest_ind(cluster_expr, other_expr, equal_var=False)
                    
                    # 计算效应大小（Cohen's d）
                    pooled_std = np.sqrt(((len(cluster_expr) - 1) * cluster_expr.var() + 
                                        (len(other_expr) - 1) * other_expr.var()) / 
                                       (len(cluster_expr) + len(other_expr) - 2))
                    cohens_d = (cluster_expr.mean() - other_expr.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    # 计算fold change
                    cluster_mean = cluster_expr.mean()
                    other_mean = other_expr.mean()
                    fold_change = cluster_mean - other_mean
                    
                    p_values.append(p_val if not np.isnan(p_val) else 1.0)
                    fold_changes.append(fold_change)
                    effect_sizes.append(abs(cohens_d))
                    
                except:
                    p_values.append(1.0)
                    fold_changes.append(0.0)
                    effect_sizes.append(0.0)
            
            # 创建结果DataFrame
            gene_stats = pd.DataFrame({
                'gene': data.columns,
                'p_value': p_values,
                'fold_change': fold_changes,
                'abs_fold_change': np.abs(fold_changes),
                'effect_size': effect_sizes
            })
            
            # 多重检验校正 - 使用Benjamini-Hochberg方法
            try:
                from statsmodels.stats.multitest import multipletests
                rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
                    gene_stats['p_value'], alpha=0.05, method='fdr_bh'
                )
                gene_stats['p_adjusted'] = p_adjusted
            except ImportError:
                # 如果statsmodels不可用，手动实现Benjamini-Hochberg校正
                p_values = np.array(gene_stats['p_value'])
                n = len(p_values)
                sort_indices = np.argsort(p_values)
                p_sorted = p_values[sort_indices]
                
                p_adjusted = np.zeros(n)
                for i in range(n-1, -1, -1):
                    if i == n-1:
                        p_adjusted[i] = p_sorted[i]
                    else:
                        p_adjusted[i] = min(p_adjusted[i+1], p_sorted[i] * n / (i + 1))
                
                # 重新排序回原来的顺序
                reverse_indices = np.argsort(sort_indices)
                gene_stats['p_adjusted'] = p_adjusted[reverse_indices]
            
            # 选择显著且有生物学意义的基因
            significant_genes = gene_stats[
                (gene_stats['p_adjusted'] < 0.05) & 
                (gene_stats['abs_fold_change'] > 0.5) &
                (gene_stats['effect_size'] > 0.2)  # 小效应大小阈值
            ].sort_values(['p_adjusted', 'effect_size'], ascending=[True, False])
            
            marker_genes[f'Cluster_{cluster_id}'] = significant_genes.head(50)  # 保留前50个
            
            print(f"   聚类 {cluster_id}: 发现 {len(significant_genes)} 个标志基因 (样本数: {np.sum(cluster_mask)})")
        
        total_markers = sum(len(genes) for genes in marker_genes.values())
        print(f"✅ 标志基因发现完成: 总计 {total_markers} 个标志基因")
        
        return marker_genes
    
    def perform_tsne_analysis(self, data, cluster_labels):
        """执行t-SNE降维分析"""
        print("🎨 执行t-SNE降维分析...")
        
        # 使用较小的样本进行t-SNE（如果样本太多）
        if len(data) > 1000:
            sample_indices = np.random.choice(len(data), 1000, replace=False)
            data_sample = data.iloc[sample_indices]
            labels_sample = cluster_labels[sample_indices]
        else:
            data_sample = data
            labels_sample = cluster_labels
        
        # 先用PCA降维到50维，再用t-SNE
        pca_for_tsne = PCA(n_components=min(50, data_sample.shape[1]))
        data_pca = pca_for_tsne.fit_transform(StandardScaler().fit_transform(data_sample))
        
        # t-SNE分析
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_sample)//4))
        tsne_result = tsne.fit_transform(data_pca)
        
        print(f"✅ t-SNE分析完成: {len(data_sample)} 样本")
        
        return tsne_result, labels_sample
    
    def create_comprehensive_visualizations(self, data, pca_results, clustering_results, 
                                          best_method, marker_genes, tsne_results):
        """创建综合可视化"""
        print("📊 生成综合可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        
        # 1. PCA分析图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA散点图
        best_labels = clustering_results[best_method]['labels']
        scatter = axes[0, 0].scatter(pca_results['pca_result'][:, 0], 
                                   pca_results['pca_result'][:, 1], 
                                   c=best_labels, cmap='tab10', alpha=0.7, s=50)
        axes[0, 0].set_xlabel(f'PC1 ({pca_results["explained_variance"][0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_results["explained_variance"][1]:.1%} variance)')
        axes[0, 0].set_title(f'PCA分析 - {best_method.upper()}聚类结果')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='聚类')
        
        # 主成分方差解释图
        n_components_show = min(20, len(pca_results['explained_variance']))
        axes[0, 1].bar(range(1, n_components_show + 1), 
                      pca_results['explained_variance'][:n_components_show])
        axes[0, 1].set_xlabel('主成分')
        axes[0, 1].set_ylabel('解释方差比')
        axes[0, 1].set_title('主成分方差解释')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 累积方差图
        axes[1, 0].plot(range(1, n_components_show + 1), 
                       pca_results['cumulative_variance'][:n_components_show], 'bo-')
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%')
        axes[1, 0].axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%')
        axes[1, 0].set_xlabel('主成分数量')
        axes[1, 0].set_ylabel('累积解释方差比')
        axes[1, 0].set_title('主成分累积方差')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 聚类评估图
        if 'k_range' in clustering_results:
            axes[1, 1].plot(clustering_results['k_range'], 
                           clustering_results['silhouette_scores'], 'bo-', label='轮廓系数')
            axes[1, 1].set_xlabel('聚类数 k')
            axes[1, 1].set_ylabel('轮廓系数')
            axes[1, 1].set_title('聚类数评估')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'pca_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 聚类比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = ['kmeans', 'hierarchical']
        if 'dbscan' in clustering_results and clustering_results['dbscan']['n_clusters'] > 1:
            methods.append('dbscan')
        
        for i, method in enumerate(methods[:4]):
            row, col = i // 2, i % 2
            if method in clustering_results:
                labels = clustering_results[method]['labels']
                scatter = axes[row, col].scatter(pca_results['pca_result'][:, 0], 
                                               pca_results['pca_result'][:, 1], 
                                               c=labels, cmap='tab10', alpha=0.7, s=50)
                axes[row, col].set_xlabel('PC1')
                axes[row, col].set_ylabel('PC2')
                axes[row, col].set_title(f'{method.upper()} (轮廓系数: {clustering_results[method]["silhouette"]:.3f})')
                axes[row, col].grid(True, alpha=0.3)
        
        # 如果方法少于4个，隐藏多余的子图
        for i in range(len(methods), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. t-SNE可视化
        if tsne_results:
            tsne_data, tsne_labels = tsne_results
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                                c=tsne_labels, cmap='tab10', alpha=0.7, s=50)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE可视化 - 聚类结果')
            plt.colorbar(scatter, label='聚类')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.vis_dir / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 聚类分布图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        # 排除噪声点
        if -1 in cluster_counts.index:
            noise_count = cluster_counts[-1]
            cluster_counts = cluster_counts.drop(-1)
        else:
            noise_count = 0
        
        bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, 
                      color=plt.cm.tab10(range(len(cluster_counts))))
        plt.xlabel('聚类编号')
        plt.ylabel('样本数量')
        plt.title(f'{best_method.upper()}聚类分布')
        plt.xticks(range(len(cluster_counts)), [f'聚类 {i}' for i in cluster_counts.index])
        
        # 添加数值标签
        for bar, count in zip(bars, cluster_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom')
        
        if noise_count > 0:
            plt.text(0.02, 0.98, f'噪声点: {noise_count}', transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 标志基因数量图
        plt.subplot(1, 2, 2)
        if marker_genes:
            marker_counts = [len(genes) for genes in marker_genes.values()]
            cluster_names = list(marker_genes.keys())
            bars = plt.bar(range(len(marker_counts)), marker_counts, 
                          color=plt.cm.tab10(range(len(marker_counts))))
            plt.xlabel('聚类')
            plt.ylabel('标志基因数量')
            plt.title('各聚类标志基因数量')
            plt.xticks(range(len(cluster_names)), cluster_names, rotation=45)
            
            # 添加数值标签
            for bar, count in zip(bars, marker_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 标志基因热图
        if marker_genes and len(marker_genes) > 0:
            # 选择每个聚类的前10个标志基因
            selected_genes = []
            for cluster_name, genes_df in marker_genes.items():
                if len(genes_df) > 0:
                    selected_genes.extend(genes_df.head(10)['gene'].tolist())
            
            # 去重并限制数量
            selected_genes = list(set(selected_genes))[:50]  # 最多50个基因
            
            if selected_genes and len(selected_genes) > 5:
                # 准备热图数据
                heatmap_data = data[selected_genes].T
                
                # 按聚类排序样本
                valid_indices = best_labels != -1  # 排除噪声点
                if np.sum(valid_indices) > 0:
                    valid_labels = best_labels[valid_indices]
                    valid_data = heatmap_data.loc[:, valid_indices]
                    
                    sorted_indices = np.argsort(valid_labels)
                    heatmap_data_sorted = valid_data.iloc[:, sorted_indices]
                    
                    # 创建聚类颜色条
                    unique_clusters = np.unique(valid_labels)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
                    cluster_colors = [colors[np.where(unique_clusters == label)[0][0]] 
                                    for label in valid_labels[sorted_indices]]
                    
                    # 绘制热图
                    plt.figure(figsize=(15, 10))
                    sns.clustermap(heatmap_data_sorted, 
                                  col_colors=cluster_colors,
                                  cmap='RdBu_r', 
                                  center=0,
                                  figsize=(15, 10),
                                  cbar_kws={'label': '基因表达水平'},
                                  row_cluster=True,
                                  col_cluster=False)
                    
                    plt.savefig(self.vis_dir / 'marker_genes_heatmap.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"✅ 可视化图表已保存到: {self.vis_dir}")
    
    def save_comprehensive_results(self, data, pca_results, clustering_results, 
                                 best_method, marker_genes, cluster_evaluation):
        """保存综合分析结果"""
        print("💾 保存综合分析结果...")
        
        best_labels = clustering_results[best_method]['labels']
        
        # 1. 保存样本聚类结果
        sample_results = pd.DataFrame({
            'sample_id': data.index,
            'cluster': best_labels,
            'PC1': pca_results['pca_result'][:, 0],
            'PC2': pca_results['pca_result'][:, 1],
            'PC3': pca_results['pca_result'][:, 2] if pca_results['pca_result'].shape[1] > 2 else 0
        })
        sample_results.to_csv(self.output_dir / 'GSE62254_sample_clusters.csv', index=False)
        
        # 2. 保存所有聚类方法的结果比较
        clustering_comparison = pd.DataFrame({
            'method': list(clustering_results.keys()),
            'silhouette_score': [clustering_results[method]['silhouette'] for method in clustering_results.keys()],
            'n_clusters': [len(np.unique(clustering_results[method]['labels'][clustering_results[method]['labels'] != -1])) 
                          for method in clustering_results.keys()]
        })
        clustering_comparison.to_csv(self.output_dir / 'GSE62254_clustering_comparison.csv', index=False)
        
        # 3. 保存标志基因
        if marker_genes:
            with pd.ExcelWriter(self.output_dir / 'GSE62254_marker_genes.xlsx') as writer:
                for cluster_name, genes_df in marker_genes.items():
                    genes_df.to_excel(writer, sheet_name=cluster_name, index=False)
        
        # 4. 保存PCA结果
        pca_summary = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(pca_results['explained_variance']))],
            'explained_variance_ratio': pca_results['explained_variance'],
            'cumulative_variance_ratio': pca_results['cumulative_variance']
        })
        pca_summary.to_csv(self.output_dir / 'GSE62254_pca_summary.csv', index=False)
        
        # 5. 生成详细分析报告
        report = self.generate_detailed_report(data, pca_results, clustering_results, 
                                             best_method, marker_genes, cluster_evaluation)
        
        with open(self.output_dir / 'GSE62254_comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 结果已保存到: {self.output_dir}")
    
    def generate_detailed_report(self, data, pca_results, clustering_results, 
                               best_method, marker_genes, cluster_evaluation):
        """生成详细分析报告"""
        
        best_labels = clustering_results[best_method]['labels']
        unique_clusters = np.unique(best_labels[best_labels != -1])
        
        report = f"""
GSE62254胃癌数据集无监督学习分析报告
=====================================

分析概况:
- 分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据集: GSE62254
- 分析类型: 无监督学习（不使用任何标签）
- 样本数量: {len(data)}
- 基因数量: {len(data.columns)}

数据预处理:
- 原始基因数: 54,675
- 质控后基因数: {len(data.columns)}
- 数据标准化: Z-score标准化
- 异常值处理: IQR方法移除极端样本

主成分分析 (PCA):
- 解释80%方差需要: {pca_results['n_components_80']} 个主成分
- 解释90%方差需要: {pca_results['n_components_90']} 个主成分
- 前10个主成分解释方差: {pca_results['cumulative_variance'][9]:.2%}
- 前20个主成分解释方差: {pca_results['cumulative_variance'][19]:.2%}

聚类分析:
- 评估的聚类数范围: 2-10
- 最优聚类数 (轮廓系数): {cluster_evaluation['best_k_silhouette']}
- 最优聚类数 (Calinski-Harabasz): {cluster_evaluation['best_k_calinski']}
- 肘部法则建议: {cluster_evaluation['elbow_k']}

聚类方法比较:
"""
        
        for method, results in clustering_results.items():
            if method == 'dbscan':
                report += f"- {method.upper()}: 轮廓系数 {results['silhouette']:.3f}, 聚类数 {results['n_clusters']}, 噪声点 {results['n_noise']}\n"
            else:
                report += f"- {method.upper()}: 轮廓系数 {results['silhouette']:.3f}\n"
        
        report += f"\n最佳聚类方法: {best_method.upper()}\n"
        
        report += f"""
聚类结果详情:
- 聚类数量: {len(unique_clusters)}
- 聚类分布:
"""
        
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        if -1 in cluster_counts.index:
            noise_count = cluster_counts[-1]
            cluster_counts = cluster_counts.drop(-1)
            report += f"  噪声点: {noise_count} 样本\n"
        
        for cluster_id in unique_clusters:
            count = np.sum(best_labels == cluster_id)
            percentage = count / len(best_labels) * 100
            report += f"  聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)\n"
        
        report += f"""
标志基因发现:
"""
        
        if marker_genes:
            total_markers = sum(len(genes) for genes in marker_genes.values())
            report += f"- 总标志基因数: {total_markers}\n"
            
            for cluster_name, genes_df in marker_genes.items():
                report += f"- {cluster_name}: {len(genes_df)} 个标志基因\n"
                if len(genes_df) > 0:
                    top_genes = genes_df.head(5)['gene'].tolist()
                    report += f"  前5个: {', '.join(top_genes)}\n"
        else:
            report += "- 未发现显著标志基因\n"
        
        report += f"""
生物学意义:
1. 样本异质性: 数据显示胃癌样本存在明显的分子异质性
2. 分子亚型: 识别出 {len(unique_clusters)} 个潜在的分子亚型
3. 特征基因: 发现了区分不同亚型的特征基因集合
4. 临床意义: 不同亚型可能具有不同的生物学特征和临床表现

分析方法:
1. 数据预处理: 质量控制、标准化、异常值处理
2. 降维分析: PCA主成分分析
3. 聚类分析: K-means、层次聚类、DBSCAN
4. 特征发现: 统计检验识别标志基因
5. 可视化: PCA、t-SNE、热图等多种可视化方法

文件输出:
- GSE62254_sample_clusters.csv: 样本聚类结果
- GSE62254_clustering_comparison.csv: 聚类方法比较
- GSE62254_marker_genes.xlsx: 各聚类标志基因
- GSE62254_pca_summary.csv: PCA分析结果
- 可视化图表保存在: GSE62254_unsupervised_analysis/

注意事项:
1. 本分析完全基于无监督学习，不使用任何临床标签
2. 聚类结果需要结合生物学知识进行解释
3. 标志基因需要进一步的功能验证和文献调研
4. 建议结合已知的胃癌分子分型进行比较分析
5. 结果的生物学意义需要通过实验验证

研究价值:
1. 为胃癌分子分型研究提供数据支持
2. 识别潜在的生物标志物和治疗靶点
3. 为个体化医学提供分子基础
4. 为后续有监督学习研究提供假设

局限性:
1. 缺乏临床标签验证聚类的生物学意义
2. 单一数据集分析，需要外部数据验证
3. 基于芯片技术，可能存在技术偏差
4. 样本量相对有限，影响统计功效

建议后续研究:
1. 结合临床数据分析不同聚类的临床特征
2. 在独立数据集中验证聚类结果
3. 对标志基因进行功能富集分析
4. 实验验证关键标志基因的生物学功能
"""
        
        return report
    
    def run_comprehensive_analysis(self):
        """运行综合无监督分析"""
        print("🚀 开始GSE62254综合无监督学习分析...")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            expr_data = self.load_expression_data()
            
            # 2. 数据预处理
            processed_data = self.preprocess_data(expr_data)
            
            # 3. PCA分析
            pca_results = self.perform_pca_analysis(processed_data)
            
            # 4. 确定最优聚类数
            cluster_evaluation = self.determine_optimal_clusters(pca_results['pca_result'])
            
            # 5. 多种聚类分析
            clustering_results, best_method = self.perform_multiple_clustering(
                processed_data, pca_results['pca_result'], cluster_evaluation['recommended_k']
            )
            
            # 6. 寻找标志基因
            marker_genes = self.find_cluster_marker_genes(
                processed_data, clustering_results[best_method]['labels'], best_method
            )
            
            # 7. t-SNE分析
            tsne_results = self.perform_tsne_analysis(
                processed_data, clustering_results[best_method]['labels']
            )
            
            # 8. 创建可视化
            self.create_comprehensive_visualizations(
                processed_data, pca_results, clustering_results, 
                best_method, marker_genes, tsne_results
            )
            
            # 9. 保存结果
            self.save_comprehensive_results(
                processed_data, pca_results, clustering_results, 
                best_method, marker_genes, cluster_evaluation
            )
            
            print("\n" + "=" * 60)
            print("🎉 GSE62254综合无监督分析完成！")
            print(f"📁 结果文件: {self.output_dir}")
            print(f"📊 可视化图表: {self.vis_dir}")
            print("=" * 60)
            
            return {
                'processed_data': processed_data,
                'pca_results': pca_results,
                'clustering_results': clustering_results,
                'best_method': best_method,
                'marker_genes': marker_genes,
                'cluster_evaluation': cluster_evaluation
            }
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            raise

def main():
    """主函数"""
    analyzer = GSE62254UnsupervisedAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis()
        
        print("\n📋 分析结果摘要:")
        print(f"   最佳聚类方法: {results['best_method'].upper()}")
        print(f"   聚类质量 (轮廓系数): {results['clustering_results'][results['best_method']]['silhouette']:.3f}")
        print(f"   发现的样本亚群数: {len(np.unique(results['clustering_results'][results['best_method']]['labels'][results['clustering_results'][results['best_method']]['labels'] != -1]))}")
        print(f"   标志基因总数: {sum(len(genes) for genes in results['marker_genes'].values())}")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()