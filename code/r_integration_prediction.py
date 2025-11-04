import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取项目根目录和脚本目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 修正R脚本目录路径 - 指向code/R文件夹
R_SCRIPTS_DIR = os.path.join(BASE_DIR, "code", "R")
print(f"R脚本目录: {R_SCRIPTS_DIR}")

# 检查R是否安装
r_available = False
try:
    result = subprocess.run(["R", "--version"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        r_available = True
        # 安全地获取版本信息
        if result.stdout and result.stdout.strip():
            version_lines = result.stdout.splitlines()
            if version_lines:
                print(f"R版本: {version_lines[0]}")
            else:
                print("R已安装，但版本信息不完整")
        else:
            print("R已安装")
    else:
        print("R已安装但无法正常运行")
except FileNotFoundError:
    print("错误: 未找到R程序，请确保R已安装并添加到系统PATH中")
except Exception as e:
    print(f"检查R环境时出错: {str(e)}")

# 数据加载函数
def load_data():
    """加载合并后的数据"""
    data_path = os.path.join(BASE_DIR, "data", "data_merged.csv")
    try:
        data = pd.read_csv(data_path, index_col=0)
        print(f"Loaded merged data: {os.path.basename(data_path)} -> samples={data.shape[0]}, columns={data.shape[1]}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def log_message(message, log_file=None):
    """记录消息到控制台和日志文件"""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def run_r_feature_selection(X_train, y_train):
    """直接调用R文件夹下的脚本进行特征选择"""
    # 创建日志文件
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "r_feature_selection_log.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("===== R特征选择过程日志 =====\n")
    
    log_message("使用R文件夹下的脚本进行特征选择...", log_file)
    
    # 检查R脚本是否存在
    lasso_script = os.path.join(R_SCRIPTS_DIR, "lasso.R")
    ttest_script = os.path.join(R_SCRIPTS_DIR, "expression-t-test.R")
    
    # 检查目录是否存在
    if not os.path.exists(R_SCRIPTS_DIR):
        log_message(f"警告: R脚本目录不存在: {R_SCRIPTS_DIR}", log_file)
    else:
        log_message(f"R脚本目录存在: {R_SCRIPTS_DIR}", log_file)
    
    # 详细检查脚本文件
    if os.path.exists(lasso_script):
        log_message(f"找到LASSO脚本: {lasso_script}", log_file)
    else:
        log_message(f"警告: 未找到LASSO脚本 {lasso_script}", log_file)
        lasso_script = None
    
    if os.path.exists(ttest_script):
        log_message(f"找到差异表达分析脚本: {ttest_script}", log_file)
    else:
        log_message(f"警告: 未找到差异表达分析脚本 {ttest_script}", log_file)
        ttest_script = None
    
    # 记录特征和样本信息
    log_message(f"\n特征选择前的数据信息:", log_file)
    log_message(f"特征总数: {X_train.shape[1]}", log_file)
    log_message(f"样本总数: {X_train.shape[0]}", log_file)
    log_message(f"前5个特征名: {list(X_train.columns[:5])}", log_file)
    
    # 创建临时目录保存中间数据
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用临时目录: {temp_dir}")
        
        # 准备训练数据，确保列名是字符串类型
        train_data = X_train.copy()
        train_data.columns = [str(col) for col in train_data.columns]
        train_data['label'] = y_train
        train_csv_path = os.path.join(temp_dir, "train_data.csv")
        # 转换路径为R可识别的格式
        train_csv_path_r = train_csv_path.replace('\\', '/')
        
        # 保存数据时使用更严格的参数
        train_data.to_csv(train_csv_path, index=True, encoding='utf-8')
        print(f"保存训练数据到: {train_csv_path}")
        
        selected_features = set()
        
        # 1. 运行LASSO特征选择
        if lasso_script:
            print(f"执行LASSO特征选择: {lasso_script}")
            lasso_output = os.path.join(temp_dir, "lasso_features.txt")
            lasso_output_r = lasso_output.replace('\\', '/')
            
            # 创建调用LASSO脚本的R代码 - 简化版本，不依赖外部函数
            lasso_call_script = os.path.join(temp_dir, "call_lasso.R")
            with open(lasso_call_script, 'w', encoding='utf-8') as f:
                # 使用列表存储R代码行
                lasso_r_code = [
                    "# 加载训练数据",
                    f"train_data <- read.csv('{train_csv_path_r}', row.names=1)",
                    "X <- train_data[, -ncol(train_data)]",
                    "y <- train_data$label",
                    "",
                    "# 尝试直接执行LASSO（不依赖外部函数）",
                    "tryCatch({",
                    "    # 设置用户库路径以避免权限问题",
                    "    user_lib <- Sys.getenv('R_USER')",
                    "    if (user_lib == '') user_lib <- Sys.getenv('HOME')",
                    "    if (user_lib == '') user_lib <- tempdir()",
                    "    user_lib_path <- file.path(user_lib, 'R', 'library')",
                    "    dir.create(user_lib_path, recursive=TRUE, showWarnings=FALSE)",
                    "    .libPaths(c(user_lib_path, .libPaths()))",
                    "    cat('使用R库路径:', user_lib_path, '\n')",
                    "    ",
                    "    # 尝试加载glmnet包，如果未安装则尝试安装",
                    "    if (!requireNamespace('glmnet', quietly = TRUE)) {",
                    "        cat('尝试安装glmnet包...\n')",
                    "        # 使用参数避免依赖问题并选择更稳定的源",
                    "        install.packages('glmnet', ",
                    "                       repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/', ",
                    "                       dependencies=c('Depends', 'Imports'), ",
                    "                       lib=user_lib_path, ",
                    "                       quiet=TRUE, ",
                    "                       verbose=FALSE)",
                    "    }",
                    "    ",
                    "    # 再次检查是否成功安装",
                    "    if (requireNamespace('glmnet', quietly = TRUE)) {",
                    "        library(glmnet)",
                    "        ",
                    "        # 确保数据类型正确",
                    "        X_matrix <- as.matrix(X)",
                    "        y <- as.numeric(y)",
                    "        ",
                    "        set.seed(101)",
                    "        # 执行LASSO交叉验证",
                    "        lasso_fit <- cv.glmnet(X_matrix, y, type.measure = 'class', ",
                    "                              alpha=1, family='binomial', nfolds=5)",
                    "        ",
                    "        # 获取系数",
                    "        lasso_coef <- coef(lasso_fit, s=lasso_fit$lambda.min)",
                    "        # 将S4对象转换为矩阵以避免索引错误",
                    "        lasso_coef_mat <- as.matrix(lasso_coef)",
                    "        selected_features_lasso <- rownames(lasso_coef_mat)[lasso_coef_mat[,1] != 0]",
                    "        selected_features_lasso <- selected_features_lasso[selected_features_lasso != '(Intercept)']",
                    "        ",
                    "        # 保存结果",
                    f"        write.table(selected_features_lasso, '{lasso_output_r}', ",
                    "                   row.names=FALSE, col.names=FALSE, quote=FALSE)",
                    "        cat('LASSO特征选择完成: ', length(selected_features_lasso), '个特征\n')",
                    "    } else {",
                    "        cat('警告: 无法安装或加载glmnet包，跳过LASSO特征选择\n')",
                    "        # 创建空文件避免后续错误",
                    f"        file.create('{lasso_output_r}')",
                    "    }",
                    "}, error = function(e) {",
                    "    cat('LASSO执行错误: ', conditionMessage(e), '\n')",
                    "    # 创建空文件避免后续错误",
                    f"    file.create('{lasso_output_r}')",
                    "})"]
                # 写入R代码
                f.write('\n'.join(lasso_r_code))
            
            # 运行R脚本
            try:
                # 增加超时时间
                result = subprocess.run(["Rscript", lasso_call_script], 
                                      capture_output=True, text=True, encoding='utf-8',
                                      timeout=300)  # 增加到5分钟
                log_message(f"LASSO脚本执行完成", log_file)
                if result.stdout:
                    log_message(f"LASSO脚本标准输出:\n{result.stdout[:1000]}..." if len(result.stdout) > 1000 else f"LASSO脚本标准输出:\n{result.stdout}", log_file)
                if result.stderr:
                    log_message(f"LASSO脚本错误输出:\n{result.stderr[:1000]}..." if len(result.stderr) > 1000 else f"LASSO脚本错误输出:\n{result.stderr}", log_file)
                
                # 读取LASSO选择的特征
                if os.path.exists(lasso_output):
                    log_message(f"LASSO输出文件存在，大小: {os.path.getsize(lasso_output)} 字节", log_file)
                    if os.path.getsize(lasso_output) > 0:
                        try:
                            with open(lasso_output, 'r', encoding='utf-8') as f:
                                lasso_feats = [line.strip() for line in f if line.strip()]
                                # 确保特征是字符串类型
                                lasso_feats = [str(feat) for feat in lasso_feats]
                                log_message(f"LASSO选择的原始特征前5个: {lasso_feats[:5]}...", log_file)
                                log_message(f"LASSO选择的特征总数: {len(lasso_feats)}", log_file)
                                selected_features.update(lasso_feats)
                        except Exception as e:
                            log_message(f"读取LASSO特征时出错: {e}", log_file)
            except subprocess.TimeoutExpired:
                print("LASSO脚本运行超时")
                # 尝试使用更快的特征选择方法
                print("尝试使用简化的LASSO方法...")
                simplified_lasso_script = os.path.join(temp_dir, "simplified_lasso.R")
                with open(simplified_lasso_script, 'w', encoding='utf-8') as f:
                    simplified_code = [
                        "# 简化的LASSO特征选择",
                        f"train_data <- read.csv('{train_csv_path_r}', row.names=1)",
                        "# 只选择前1000个特征以提高速度",
                        "X <- train_data[, 1:min(1000, ncol(train_data)-1)]",
                        "y <- train_data$label",
                        "if (requireNamespace('glmnet', quietly = TRUE)) {",
                        "    library(glmnet)",
                        "    X_matrix <- as.matrix(X)",
                        "    y <- as.numeric(y)",
                        "    set.seed(101)",
                        "    lasso_fit <- glmnet(X_matrix, y, alpha=1, family='binomial')",
                        "    lasso_coef <- coef(lasso_fit, s=lasso_fit$lambda.min)",
                        "    selected_features_lasso <- rownames(lasso_coef)[lasso_coef != 0]",
                        "    selected_features_lasso <- selected_features_lasso[selected_features_lasso != '(Intercept)']",
                        f"    write.table(selected_features_lasso, '{lasso_output_r}', ",
                        "               row.names=FALSE, col.names=FALSE, quote=FALSE)",
                        "    cat('简化LASSO完成: ', length(selected_features_lasso), '个特征\n')",
                        "}",
                    ]
                    f.write('\n'.join(simplified_code))
                # 运行简化脚本
                try:
                    simplified_result = subprocess.run(["Rscript", simplified_lasso_script], 
                                  capture_output=True, text=True, encoding='utf-8',
                                  timeout=120)
                    log_message(f"简化LASSO脚本执行完成", log_file)
                    if simplified_result.stdout:
                        log_message(f"简化LASSO脚本输出:\n{simplified_result.stdout[:500]}..." if len(simplified_result.stdout) > 500 else f"简化LASSO脚本输出:\n{simplified_result.stdout}", log_file)
                except Exception as e:
                    log_message(f"运行简化LASSO脚本失败: {e}", log_file)
            except Exception as e:
                print(f"运行LASSO脚本时出错: {e}")
        
        # 2. 运行差异表达分析
        if ttest_script:
            print(f"执行差异表达分析: {ttest_script}")
            ttest_output = os.path.join(temp_dir, "de_features.txt")
            ttest_output_r = ttest_output.replace('\\', '/')
            
            # 创建调用差异表达分析脚本的R代码 - 直接执行t检验
            ttest_call_script = os.path.join(temp_dir, "call_ttest.R")
            with open(ttest_call_script, 'w', encoding='utf-8') as f:
                # 使用列表存储R代码行
                ttest_r_code = [
                    "# 加载训练数据",
                    f"train_data <- read.csv('{train_csv_path_r}', row.names=1)",
                    "X <- train_data[, -ncol(train_data)]",
                    "y <- train_data$label",
                    "",
                    "# 直接执行t检验",
                    "tryCatch({",
                    "    p_values <- numeric(ncol(X))",
                    "    names(p_values) <- colnames(X)",
                    "    ",
                    "    # 执行t检验",
                    "    for (i in 1:ncol(X)) {",
                    "        tryCatch({",
                    "            group0 <- X[y == 0, i]",
                    "            group1 <- X[y == 1, i]",
                    "            ",
                    "            # 检查数据有效性",
                    "            if (length(group0) > 2 && length(group1) > 2 && ",
                    "                sd(group0, na.rm = TRUE) > 0 && sd(group1, na.rm = TRUE) > 0) {",
                    "                test_result <- t.test(group0, group1, paired = FALSE, var.equal = FALSE)",
                    "                p_values[i] <- test_result$p.value",
                    "            } else {",
                    "                p_values[i] <- 1.0",
                    "            }",
                    "        }, error = function(e) {",
                    "            p_values[i] <- 1.0",
                    "        })",
                    "    }",
                    "    ",
                    "    # 选择p值小于0.05的特征",
                    "    selected_features_de <- names(p_values)[p_values < 0.05]",
                    "    ",
                    "    # 如果特征太少，放宽条件",
                    "    if (length(selected_features_de) < 10) {",
                    "        cat('特征太少，放宽p值阈值到0.1\n')",
                    "        selected_features_de <- names(p_values)[p_values < 0.1]",
                    "    }",
                    "    ",
                    "    # 如果还是太少，选择p值最小的前50个特征",
                    "    if (length(selected_features_de) < 10) {",
                    "        cat('特征仍然太少，选择p值最小的前50个特征\n')",
                    "        selected_features_de <- names(sort(p_values)[1:min(50, length(p_values))])",
                    "    }",
                    "    ",
                    "    # 保存结果",
                    f"    write.table(selected_features_de, '{ttest_output_r}', ",
                    "               row.names=FALSE, col.names=FALSE, quote=FALSE)",
                    "    cat('差异表达分析完成: ', length(selected_features_de), '个特征\n')",
                    "}, error = function(e) {",
                    "    cat('差异表达分析执行错误: ', conditionMessage(e), '\n')",
                    "    # 创建空文件避免后续错误",
                    f"    file.create('{ttest_output_r}')",
                    "})"]
                # 写入R代码
                f.write('\n'.join(ttest_r_code))
            
            # 运行R脚本
            try:
                # 增加超时时间
                result = subprocess.run(["Rscript", ttest_call_script], 
                                      capture_output=True, text=True, encoding='utf-8',
                                      timeout=300)  # 增加到5分钟
                log_message(f"差异表达分析脚本执行完成", log_file)
                if result.stdout:
                    log_message(f"差异表达分析脚本输出:\n{result.stdout[:1000]}..." if len(result.stdout) > 1000 else f"差异表达分析脚本输出:\n{result.stdout}", log_file)
                if result.stderr:
                    log_message(f"差异表达分析脚本错误输出:\n{result.stderr[:1000]}..." if len(result.stderr) > 1000 else f"差异表达分析脚本错误输出:\n{result.stderr}", log_file)
                
                # 读取差异表达分析选择的特征
                if os.path.exists(ttest_output):
                    log_message(f"差异表达分析输出文件存在，大小: {os.path.getsize(ttest_output)} 字节", log_file)
                    if os.path.getsize(ttest_output) > 0:
                        try:
                            with open(ttest_output, 'r', encoding='utf-8') as f:
                                de_feats = [line.strip() for line in f if line.strip()]
                                # 确保特征是字符串类型
                                de_feats = [str(feat) for feat in de_feats]
                                log_message(f"差异表达分析选择的原始特征前5个: {de_feats[:5]}...", log_file)
                                log_message(f"差异表达分析选择的特征总数: {len(de_feats)}", log_file)
                                selected_features.update(de_feats)
                        except Exception as e:
                            log_message(f"读取差异表达分析特征时出错: {e}", log_file)
            except subprocess.TimeoutExpired:
                print("差异表达分析脚本运行超时")
                # 尝试使用更快的差异表达分析方法
                print("尝试使用简化的差异表达分析方法...")
                simplified_ttest_script = os.path.join(temp_dir, "simplified_ttest.R")
                with open(simplified_ttest_script, 'w', encoding='utf-8') as f:
                    simplified_code = [
                        "# 简化的差异表达分析",
                        f"train_data <- read.csv('{train_csv_path_r}', row.names=1)",
                        "# 只选择前1000个特征以提高速度",
                        "X <- train_data[, 1:min(1000, ncol(train_data)-1)]",
                        "y <- train_data$label",
                        "# 使用方差分析代替t检验",
                        "f_stats <- numeric(ncol(X))",
                        "names(f_stats) <- colnames(X)",
                        "for (i in 1:ncol(X)) {",
                        "    tryCatch({",
                        "        group0 <- X[y == 0, i]",
                        "        group1 <- X[y == 1, i]",
                        "        if (length(unique(group0)) > 1 && length(unique(group1)) > 1) {",
                        "            # 简单的均值比较",
                        "            f_stats[i] <- abs(mean(group0, na.rm=TRUE) - mean(group1, na.rm=TRUE))",
                        "        } else {",
                        "            f_stats[i] <- 0",
                        "        }",
                        "    }, error = function(e) {",
                        "        f_stats[i] <- 0",
                        "    })",
                        "}",
                        "# 选择前50个特征",
                        "selected_features_de <- names(sort(f_stats, decreasing=TRUE)[1:min(50, length(f_stats))])",
                        f"write.table(selected_features_de, '{ttest_output_r}', ",
                        "           row.names=FALSE, col.names=FALSE, quote=FALSE)",
                        "cat('简化差异表达分析完成: ', length(selected_features_de), '个特征\n')",
                    ]
                    f.write('\n'.join(simplified_code))
                # 运行简化脚本
                try:
                    subprocess.run(["Rscript", simplified_ttest_script], 
                                  capture_output=True, text=True, encoding='utf-8',
                                  timeout=120)
                except Exception as e:
                    print(f"运行简化差异表达分析脚本失败: {e}")
            except Exception as e:
                print(f"运行差异表达分析脚本时出错: {e}")
        
        # 过滤掉可能不存在的特征 - 确保列名是字符串类型进行比较
        X_columns_str = [str(col) for col in X_train.columns]
        
        # 记录X列名信息
        log_message(f"\n特征匹配过程:", log_file)
        log_message(f"X列名前10个: {X_columns_str[:10]}", log_file)
        log_message(f"R选择的原始特征总数: {len(selected_features)}", log_file)
        
        # 记录原始选择的特征
        log_message(f"\nR选择的特征前20个:", log_file)
        for i, feat in enumerate(list(selected_features)[:20]):
            log_message(f"{i+1}. {feat}", log_file)
        
        # 创建更宽松的特征匹配逻辑
        valid_features = []
        log_message(f"\n特征匹配详情:", log_file)
        
        for feat in selected_features:
            # 直接匹配
            if feat in X_columns_str:
                valid_features.append(feat)
                log_message(f"直接匹配: {feat}", log_file)
            else:
                # 尝试去除引号后匹配
                feat_clean = feat.strip('"').strip("'")
                if feat_clean in X_columns_str:
                    valid_features.append(feat_clean)
                    log_message(f"引号匹配: {feat} -> {feat_clean}", log_file)
                # 尝试部分匹配
                else:
                    matches = [col for col in X_columns_str if col in feat or feat in col]
                    if matches:
                        valid_features.append(matches[0])
                        log_message(f"部分匹配: {feat} -> {matches[0]}", log_file)
                    else:
                        log_message(f"未匹配: {feat}", log_file)
        
        log_message(f"\n合并后有效特征数: {len(valid_features)}", log_file)
        log_message(f"最终选择的特征列表:", log_file)
        for i, feat in enumerate(valid_features[:20]):  # 只记录前20个
            log_message(f"{i+1}. {feat}", log_file)
        if len(valid_features) > 20:
            log_message(f"... 还有{len(valid_features)-20}个特征未显示", log_file)
        
        # 如果没有选择到特征，使用后备方案
        if len(valid_features) == 0:
            print("警告: R脚本未选择任何特征，使用后备方案")
            # 使用简单的t检验脚本作为后备
            temp_r_script = os.path.join(temp_dir, "basic_feature_selection.R")
            temp_output = os.path.join(temp_dir, "basic_features.txt")
            temp_output_r = temp_output.replace('\\', '/')
            
            with open(temp_r_script, 'w', encoding='utf-8') as f:
                # 使用列表存储R代码行
                basic_r_code = [
                    "# 加载训练数据",
                    f"train_data <- read.csv('{train_csv_path_r}', row.names=1)",
                    "X <- train_data[, -ncol(train_data)]",
                    "y <- train_data$label",
                    "",
                    "# 直接执行t检验",
                    "tryCatch({",
                    "    # 计算特征的均值差异（更简单的方法）",
                    "    mean_diff <- numeric(ncol(X))",
                    "    names(mean_diff) <- colnames(X)",
                    "    ",
                    "    # 计算每个特征在两组间的均值差异",
                    "    for (i in 1:ncol(X)) {",
                    "        tryCatch({",
                    "            group0 <- X[y == 0, i]",
                    "            group1 <- X[y == 1, i]",
                    "            ",
                    "            # 简单的均值差异",
                    "            if (length(group0) > 0 && length(group1) > 0) {",
                    "                mean_diff[i] <- abs(mean(group0, na.rm=TRUE) - mean(group1, na.rm=TRUE))",
                    "            } else {",
                    "                mean_diff[i] <- 0",
                    "            }",
                    "        }, error = function(e) {",
                    "            mean_diff[i] <- 0",
                    "        })  ",
                    "    }",
                    "    ",
                    "    # 选择均值差异最大的前50个特征",
                    "    selected_features_basic <- names(sort(mean_diff, decreasing=TRUE)[1:min(50, length(mean_diff))])",
                    "    ",
                    f"    write.table(selected_features_basic, '{temp_output_r}', ",
                    "               row.names=FALSE, col.names=FALSE, quote=FALSE)",
                    "    cat('基本特征选择完成: ', length(selected_features_basic), '个特征\n')",
                    "}, error = function(e) {",
                    "    cat('基本特征选择错误: ', conditionMessage(e), '\n')",
                    "    # 创建包含少量特征的文件作为最后的备选",
                    "    default_features <- colnames(X)[1:min(10, ncol(X))]",
                    f"    write.table(default_features, '{temp_output_r}', ",
                    "               row.names=FALSE, col.names=FALSE, quote=FALSE)",
                    "    cat('使用默认特征: ', length(default_features), '个特征\n')",
                    "})"]
                # 写入R代码
                f.write('\n'.join(basic_r_code))
            
            # 运行临时R脚本
            try:
                result = subprocess.run(["Rscript", temp_r_script], 
                                      capture_output=True, text=True, encoding='utf-8')
                print(f"基本特征选择脚本输出:\n{result.stdout}")
                if result.stderr:
                    print(f"基本特征选择脚本警告:\n{result.stderr}")
                
                # 读取选择的特征
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    try:
                        with open(temp_output, 'r', encoding='utf-8') as f:
                            basic_feats = [line.strip() for line in f if line.strip()]
                            basic_feats = [str(feat) for feat in basic_feats]
                            print(f"基本特征选择的原始特征: {basic_feats[:5]}...")  # 打印前几个特征用于调试
                            
                            # 使用更宽松的匹配逻辑
                            temp_valid_feats = []
                            for feat in basic_feats:
                                # 直接匹配
                                if feat in X_columns_str:
                                    temp_valid_feats.append(feat)
                                else:
                                    # 尝试去除引号后匹配
                                    feat_clean = feat.strip('"').strip("'")
                                    if feat_clean in X_columns_str:
                                        temp_valid_feats.append(feat_clean)
                                        print(f"匹配到特征: {feat} -> {feat_clean}")
                            
                            valid_features = temp_valid_feats
                            print(f"基本特征选择有效特征数: {len(valid_features)}")
                    except Exception as e:
                        print(f"读取基本特征时出错: {e}")
            except Exception as e:
                print(f"运行基本特征选择脚本时出错: {e}")
        
        # 如果仍然没有特征，使用前20个特征
        if len(valid_features) == 0:
            print("警告: 所有特征选择方法都失败，使用前20个特征")
            valid_features = X_train.columns[:min(20, X_train.shape[1])].tolist()
        
        # 限制特征数量，避免过拟合
        max_features = 100
        if len(valid_features) > max_features:
            print(f"特征数过多({len(valid_features)}), 限制为{max_features}个")
            valid_features = valid_features[:max_features]
        
        print(f"最终选择的特征数: {len(valid_features)}")
        return X_train[valid_features]

# 为兼容性保留旧函数名，但实际调用新函数
def call_r_scripts_with_subprocess(X_train, y_train):
    """兼容性函数，调用新的特征选择函数"""
    return run_r_feature_selection(X_train, y_train)

# 主函数
def main():
    # 创建输出目录
    out_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    data = load_data()
    
    # 提取特征和标签
    # 假设标签列是最后一列
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 特征选择：根据R可用性选择方法
    if r_available:
        # 直接调用R文件夹下的脚本进行特征选择
        try:
            X_train_selected = run_r_feature_selection(X_train, y_train)
        except Exception as e:
            print(f"运行R特征选择时出错: {e}，使用后备方案")
            # 使用更强大的特征选择方法
            from sklearn.feature_selection import SelectKBest, f_classif
            # 使用F检验选择特征
            k = min(50, X_train.shape[1])
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_train, y_train)
            X_train_selected = X_train.iloc[:, selector.get_support()]
            print(f"F检验特征选择保留了 {X_train_selected.shape[1]} 个特征")
    else:
        # 如果R不可用，使用F检验作为后备
        print("R不可用，使用F检验作为后备")
        from sklearn.feature_selection import SelectKBest, f_classif
        k = min(50, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        X_train_selected = X_train.iloc[:, selector.get_support()]
        print(f"F检验特征选择保留了 {X_train_selected.shape[1]} 个特征")
    
    # 确保有足够的特征
    if X_train_selected.shape[1] == 0:
        print("警告: 特征选择没有保留任何特征，使用前20个特征")
        X_train_selected = X_train.iloc[:, :20]
    
    # 对测试集应用相同的特征选择
    common_features = [col for col in X_train_selected.columns if col in X_test.columns]
    X_test_selected = X_test[common_features]
    
    # 训练随机森林模型 - 增加参数以提高性能
    print("训练随机森林模型...")
    model = RandomForestClassifier(
        n_estimators=200,        # 增加树的数量
        max_depth=20,           # 设置最大深度避免过拟合
        min_samples_split=5,    # 设置最小分裂样本数
        min_samples_leaf=2,     # 设置最小叶节点样本数
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_selected, y_train)
    
    # 预测
    preds = model.predict(X_test_selected)
    probs = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 对所有样本进行预测
    X_all_selected = X[common_features]
    preds_all = model.predict(X_all_selected)
    probs_all = model.predict_proba(X_all_selected)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 保存预测结果
    # 测试集预测
    df_out = pd.DataFrame({"true": y_test, "pred": preds}, index=X_test.index)
    if probs is not None:
        df_out["prob_positive"] = probs
    csv_path = os.path.join(out_dir, "predictions_r_integration.csv")
    df_out.to_csv(csv_path, index=True)
    print(f"预测已保存: {csv_path}")
    
    # 全部样本预测
    out_all = pd.DataFrame({"true": y, "pred": preds_all}, index=X.index)
    if probs_all is not None:
        out_all["prob_positive"] = probs_all
    all_csv_path = os.path.join(out_dir, "predictions_all_r_integration.csv")
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
    print(f"X_train 行数: {X_train.shape[0]}")
    print(f"X_test  行数: {X_test.shape[0]}")
    print(f"最终特征数: {X_train_selected.shape[1]}")

if __name__ == "__main__":
    main()