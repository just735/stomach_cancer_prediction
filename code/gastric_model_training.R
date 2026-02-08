options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  if (requireNamespace("caret", quietly = TRUE)) {
    library(caret)
  }
  if (requireNamespace("pROC", quietly = TRUE)) {
    library(pROC)
  }
})

base_dir <- normalizePath(getwd())
if (!dir.exists(file.path(base_dir, "data"))) {
  base_dir <- normalizePath(file.path(base_dir, ".."))
}
processed_dir <- file.path(base_dir, "data", "processed_gastric", "r_pipeline")
out_dir <- file.path(base_dir, "output", "r_pipeline", "model_training")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

expr_path <- file.path(processed_dir, "biomarker_expression.csv")
meta_path <- file.path(processed_dir, "integrated_metadata.csv")
if (!file.exists(expr_path) || !file.exists(meta_path)) {
  stop("缺少 biomarker_expression.csv 或 integrated_metadata.csv")
}

expr <- read.csv(expr_path, check.names = FALSE)
if (ncol(expr) < 2) {
  stop("biomarker_expression.csv 列数不足")
}
if (colnames(expr)[1] == "" || colnames(expr)[1] == "X") {
  rownames(expr) <- as.character(expr[[1]])
  expr <- expr[, -1, drop = FALSE]
} else {
  rownames(expr) <- as.character(expr[[1]])
  expr <- expr[, -1, drop = FALSE]
}
rn <- rownames(expr)
expr <- as.data.frame(lapply(expr, function(x) suppressWarnings(as.numeric(x))), check.names = FALSE)
rownames(expr) <- rn

meta <- read.csv(meta_path, check.names = FALSE)
meta$sample_id <- as.character(meta$sample_id)
common <- intersect(rownames(expr), meta$sample_id)
if (length(common) < 4) {
  stop("可用于训练的样本不足")
}
expr <- expr[common, , drop = FALSE]
meta <- meta[match(common, meta$sample_id), , drop = FALSE]
labels <- meta$metastasis
keep <- !is.na(labels)
expr <- expr[keep, , drop = FALSE]
labels <- labels[keep]
labels <- factor(ifelse(labels == 1, "M1", "M0"), levels = c("M0", "M1"))
if (length(unique(labels)) < 2) {
  stop("转移标签不足两类")
}

set.seed(42)
if (requireNamespace("caret", quietly = TRUE)) {
  idx <- createDataPartition(labels, p = 0.8, list = FALSE)
} else {
  idx <- unlist(tapply(seq_along(labels), labels, function(x) {
    n <- max(1, floor(0.8 * length(x)))
    sample(x, n)
  }))
}
train_x <- expr[idx, , drop = FALSE]
test_x <- expr[-idx, , drop = FALSE]
train_y <- labels[idx]
test_y <- labels[-idx]

train_x <- scale(train_x)
test_x <- scale(test_x, center = attr(train_x, "scaled:center"), scale = attr(train_x, "scaled:scale"))

train_df <- as.data.frame(train_x)
train_df$label <- train_y
test_df <- as.data.frame(test_x)
test_df$label <- test_y

safe_train <- function(formula, data, method, tuneGrid = NULL, family = NULL) {
  if (!requireNamespace("caret", quietly = TRUE)) {
    return(NULL)
  }
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = defaultSummary)
  tryCatch({
    tf <- tempfile()
    sink(tf)
    sink(tf, type = "message")
    on.exit({
      sink(type = "message")
      sink()
      unlink(tf)
    }, add = TRUE)
    suppressWarnings(suppressMessages(
      train(formula, data = data, method = method, metric = "Accuracy", trControl = ctrl, tuneGrid = tuneGrid, family = family)
    ))
  }, error = function(e) NULL)
}

results <- data.frame(model = character(), accuracy = numeric(), f1 = numeric(), auc = numeric(), status = character(), stringsAsFactors = FALSE)

rf_grid_search <- function(train_x, train_y) {
  if (!requireNamespace("randomForest", quietly = TRUE) || !requireNamespace("pROC", quietly = TRUE)) {
    return(NULL)
  }
  p <- ncol(train_x)
  mtry_vals <- unique(pmax(1, pmin(p, c(floor(log2(p)), 16, 21, floor(sqrt(p))))))
  ntree_vals <- seq(60, 95, 5)
  maxnodes_vals <- 2:7
  if (requireNamespace("caret", quietly = TRUE)) {
    folds <- caret::createFolds(train_y, k = 5, returnTrain = FALSE)
  } else {
    set.seed(1412)
    idx <- sample(seq_len(nrow(train_x)))
    folds <- split(idx, rep(1:5, length.out = length(idx)))
  }
  best_auc <- -Inf
  best_params <- NULL
  for (mtry in mtry_vals) {
    for (ntree in ntree_vals) {
      for (maxnodes in maxnodes_vals) {
        aucs <- c()
        for (fold in folds) {
          test_idx <- fold
          train_idx <- setdiff(seq_len(nrow(train_x)), test_idx)
          rf <- randomForest::randomForest(x = train_x[train_idx, , drop = FALSE], y = train_y[train_idx], ntree = ntree, mtry = mtry, maxnodes = maxnodes)
          prob <- predict(rf, newdata = train_x[test_idx, , drop = FALSE], type = "prob")[, "M1"]
          roc_obj <- pROC::roc(response = train_y[test_idx], predictor = prob, levels = c("M0", "M1"), direction = "<")
          aucs <- c(aucs, as.numeric(pROC::auc(roc_obj)))
        }
        mean_auc <- mean(aucs, na.rm = TRUE)
        if (mean_auc > best_auc) {
          best_auc <- mean_auc
          best_params <- list(mtry = mtry, ntree = ntree, maxnodes = maxnodes)
        }
      }
    }
  }
  if (is.null(best_params)) return(NULL)
  model <- randomForest::randomForest(x = train_x, y = train_y, ntree = best_params$ntree, mtry = best_params$mtry, maxnodes = best_params$maxnodes)
  list(model = model, cv_auc = best_auc, params = best_params)
}

predict_with_prob <- function(model, test_df) {
  test_x <- test_df[, setdiff(names(test_df), "label"), drop = FALSE]
  if (inherits(model, "glm")) {
    prob <- as.numeric(predict(model, newdata = test_x, type = "response"))
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "cv.glmnet") || inherits(model, "glmnet")) {
    prob <- as.numeric(predict(model, newx = as.matrix(test_x), s = "lambda.min", type = "response"))
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "randomForest")) {
    prob <- predict(model, newdata = test_x, type = "prob")[, "M1"]
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "ksvm")) {
    prob <- predict(model, as.matrix(test_x), type = "probabilities")[, "M1"]
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "svm")) {
    p <- predict(model, newdata = test_x, probability = TRUE)
    prob <- attr(p, "probabilities")[, "M1"]
    if (is.null(prob)) {
      prob <- rep(NA_real_, nrow(test_x))
    }
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "gbm")) {
    prob <- predict(model, newdata = test_x, n.trees = model$n.trees, type = "response")
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "xgb.Booster")) {
    prob <- as.numeric(predict(model, newdata = as.matrix(test_x)))
    pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
    return(list(pred = pred, prob = prob))
  }
  if (inherits(model, "train")) {
    pred <- tryCatch(predict(model, newdata = test_df), error = function(e) NULL)
    if (is.null(pred)) return(NULL)
    prob <- tryCatch(predict(model, newdata = test_df, type = "prob")[, "M1"], error = function(e) rep(NA_real_, nrow(test_df)))
    return(list(pred = pred, prob = prob))
  }
  NULL
}

eval_model <- function(name, model, test_df) {
  if (is.null(model)) {
    results <<- rbind(results, data.frame(model = name, accuracy = NA_real_, f1 = NA_real_, auc = NA_real_, status = "skip"))
    return(NULL)
  }
  out <- predict_with_prob(model, test_df)
  if (is.null(out)) {
    results <<- rbind(results, data.frame(model = name, accuracy = NA_real_, f1 = NA_real_, auc = NA_real_, status = "predict_fail"))
    return(NULL)
  }
  pred <- out$pred
  prob <- out$prob
  acc <- mean(pred == test_df$label)
  if (requireNamespace("caret", quietly = TRUE)) {
    f1 <- caret::F_meas(pred, test_df$label, positive = "M1")
  } else {
    tp <- sum(pred == "M1" & test_df$label == "M1")
    fp <- sum(pred == "M1" & test_df$label == "M0")
    fn <- sum(pred == "M0" & test_df$label == "M1")
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  }
  auc_val <- NA_real_
  if (requireNamespace("pROC", quietly = TRUE) && !all(is.na(prob))) {
    roc_obj <- tryCatch(pROC::roc(response = test_df$label, predictor = prob, levels = c("M0", "M1"), direction = "<"), error = function(e) NULL)
    if (!is.null(roc_obj)) auc_val <- as.numeric(pROC::auc(roc_obj))
  }
  results <<- rbind(results, data.frame(model = name, accuracy = acc, f1 = f1, auc = auc_val, status = "ok"))
  saveRDS(model, file.path(out_dir, paste0("model_", name, ".rds")))
  NULL
}

lr_model <- NULL
if (requireNamespace("glmnet", quietly = TRUE)) {
  y_bin <- as.integer(train_y == "M1")
  lr_model <- glmnet::cv.glmnet(as.matrix(train_x), y_bin, family = "binomial", alpha = 0, nfolds = 5)
} else if (requireNamespace("caret", quietly = TRUE) && requireNamespace("glmnet", quietly = TRUE)) {
  lr_model <- safe_train(label ~ ., train_df, "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = 0.01))
} else {
  lr_model <- suppressWarnings(glm(label ~ ., data = train_df, family = binomial()))
}
invisible(eval_model("LR", lr_model, test_df))

if (requireNamespace("caret", quietly = TRUE) && requireNamespace("randomForest", quietly = TRUE)) {
  rf_model <- safe_train(label ~ ., train_df, "rf", tuneGrid = expand.grid(mtry = c(2, 4, 8)))
  if (is.null(rf_model)) {
    rf_search <- rf_grid_search(train_x, train_y)
    if (!is.null(rf_search)) {
      rf_model <- rf_search$model
    } else {
      rf_model <- randomForest::randomForest(x = train_x, y = train_y)
    }
  }
  invisible(eval_model("RF", rf_model, test_df))
} else {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    rf_search <- rf_grid_search(train_x, train_y)
    if (!is.null(rf_search)) {
      rf_model <- rf_search$model
    } else {
      rf_model <- randomForest::randomForest(x = train_x, y = train_y)
    }
    invisible(eval_model("RF", rf_model, test_df))
  } else {
    invisible(eval_model("RF", NULL, test_df))
  }
}

if (requireNamespace("caret", quietly = TRUE) && requireNamespace("kernlab", quietly = TRUE)) {
  svm_model <- safe_train(label ~ ., train_df, "svmRadial", tuneGrid = expand.grid(C = c(0.5, 1, 2), sigma = c(0.01, 0.05)))
  if (is.null(svm_model)) {
    svm_model <- kernlab::ksvm(as.matrix(train_x), train_y, type = "C-svc", kernel = "rbfdot", prob.model = TRUE)
  }
  invisible(eval_model("SVM", svm_model, test_df))
} else {
  if (requireNamespace("kernlab", quietly = TRUE)) {
    svm_model <- kernlab::ksvm(as.matrix(train_x), train_y, type = "C-svc", kernel = "rbfdot", prob.model = TRUE)
    invisible(eval_model("SVM", svm_model, test_df))
  } else if (requireNamespace("e1071", quietly = TRUE)) {
    svm_model <- e1071::svm(x = train_x, y = train_y, type = "C-classification", kernel = "radial", probability = TRUE)
    invisible(eval_model("SVM", svm_model, test_df))
  } else {
    invisible(eval_model("SVM", NULL, test_df))
  }
}

if (requireNamespace("caret", quietly = TRUE) && requireNamespace("gbm", quietly = TRUE)) {
  gbdt_model <- safe_train(label ~ ., train_df, "gbm", tuneGrid = expand.grid(interaction.depth = c(2, 3, 4), n.trees = c(100, 200), shrinkage = c(0.05, 0.1), n.minobsinnode = 10))
  if (is.null(gbdt_model)) {
    gbm_df <- train_df
    gbm_df$label <- ifelse(gbm_df$label == "M1", 1, 0)
    gbdt_model <- gbm::gbm(label ~ ., data = gbm_df, distribution = "bernoulli", n.trees = 200, interaction.depth = 3, shrinkage = 0.05, n.minobsinnode = 10, verbose = FALSE)
  }
  invisible(eval_model("GBDT", gbdt_model, test_df))
} else {
  if (requireNamespace("gbm", quietly = TRUE)) {
    gbm_df <- train_df
    gbm_df$label <- ifelse(gbm_df$label == "M1", 1, 0)
    gbdt_model <- gbm::gbm(label ~ ., data = gbm_df, distribution = "bernoulli", n.trees = 200, interaction.depth = 3, shrinkage = 0.05, n.minobsinnode = 10, verbose = FALSE)
    invisible(eval_model("GBDT", gbdt_model, test_df))
  } else {
    invisible(eval_model("GBDT", NULL, test_df))
  }
}

if (requireNamespace("caret", quietly = TRUE) && requireNamespace("xgboost", quietly = TRUE)) {
  xgb_model <- safe_train(label ~ ., train_df, "xgbTree", tuneGrid = expand.grid(nrounds = c(100, 200), max_depth = c(3, 5), eta = c(0.05, 0.1), gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8))
  if (is.null(xgb_model)) {
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(train_x), label = as.integer(train_y == "M1"))
    xgb_model <- xgboost::xgb.train(
      params = list(objective = "binary:logistic", max_depth = 5, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8),
      data = dtrain,
      nrounds = 200,
      verbose = 0
    )
  }
  invisible(eval_model("XGBoost", xgb_model, test_df))
} else {
  if (requireNamespace("xgboost", quietly = TRUE)) {
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(train_x), label = as.integer(train_y == "M1"))
    xgb_model <- xgboost::xgb.train(
      params = list(objective = "binary:logistic", max_depth = 5, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8),
      data = dtrain,
      nrounds = 200,
      verbose = 0
    )
    invisible(eval_model("XGBoost", xgb_model, test_df))
  } else {
    invisible(eval_model("XGBoost", NULL, test_df))
  }
}

write.csv(results, file.path(out_dir, "model_performance.csv"), row.names = FALSE)
print(results)
cat("模型评估结果已保存:", file.path(out_dir, "model_performance.csv"), "\n")
