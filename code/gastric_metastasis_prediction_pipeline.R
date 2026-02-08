options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  if (requireNamespace("GEOquery", quietly = TRUE)) {
    library(GEOquery)
  }
  library(limma)
  library(WGCNA)
  library(glmnet)
  library(sva)
  library(clusterProfiler)
  library(caret)
  library(pROC)
})

base_dir <- normalizePath(getwd())
if (!dir.exists(file.path(base_dir, "data"))) {
  base_dir <- normalizePath(file.path(base_dir, ".."))
}
if (!dir.exists(file.path(base_dir, "data"))) {
  stop("未找到 data 目录")
}

geo_dir <- file.path(base_dir, "data", "GEO")
stad_dir <- file.path(base_dir, "data", "STAD", "clinical.project-tcga-stad.2026-01-25")
output_dir <- file.path(base_dir, "output", "r_pipeline")
processed_dir <- file.path(base_dir, "data", "processed_gastric", "r_pipeline")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

gse_ids <- c("GSE15459", "GSE62254", "GSE84437", "GSE26901", "GSE159929")

clean_symbol <- function(x) {
  x <- as.character(x)
  x <- gsub("\"", "", x)
  x <- gsub("///", ";", x)
  x <- gsub("//", ";", x)
  x <- gsub("\\|", ";", x)
  x <- gsub(",", ";", x)
  parts <- strsplit(x, ";", fixed = TRUE)
  vapply(parts, function(p) {
    p <- trimws(p)
    p <- p[p != ""]
    if (length(p) == 0) NA_character_ else p[1]
  }, character(1))
}

map_probes_to_symbols <- function(expr, fdata) {
  fcols <- tolower(colnames(fdata))
  sym_col <- NULL
  for (c in fcols) {
    if (grepl("symbol", c) || grepl("gene", c) && grepl("symbol|name", c)) {
      sym_col <- c
      break
    }
  }
  if (is.null(sym_col)) {
    return(expr)
  }
  symbols <- clean_symbol(fdata[[sym_col]])
  keep <- !is.na(symbols) & symbols != ""
  expr <- expr[keep, , drop = FALSE]
  symbols <- symbols[keep]
  expr_df <- as.data.frame(expr)
  expr_df$gene_symbol <- symbols
  aggregated <- aggregate(. ~ gene_symbol, data = expr_df, FUN = mean)
  rownames(aggregated) <- aggregated$gene_symbol
  aggregated$gene_symbol <- NULL
  as.matrix(aggregated)
}

parse_geo_metadata <- function(lines, sample_ids) {
  meta <- data.frame(row.names = sample_ids)
  for (line in lines) {
    if (!startsWith(line, "!Sample_")) next
    parts <- strsplit(line, "\t", fixed = TRUE)[[1]]
    key <- tolower(sub("^!Sample_", "", parts[1]))
    values <- gsub("\"", "", parts[-1])
    if (length(values) != length(sample_ids)) next
    if (key == "characteristics_ch1") {
      for (i in seq_along(values)) {
        raw <- values[i]
        if (grepl(":", raw, fixed = TRUE)) {
          kv <- strsplit(raw, ":", fixed = TRUE)[[1]]
          k <- tolower(gsub(" ", "_", trimws(kv[1])))
          v <- trimws(paste(kv[-1], collapse = ":"))
          if (k %in% colnames(meta) && !is.na(meta[i, k]) && meta[i, k] != "") {
            if (!grepl(v, meta[i, k], fixed = TRUE)) {
              meta[i, k] <- paste(meta[i, k], v, sep = ";")
            }
          } else {
            meta[i, k] <- v
          }
        } else {
          meta[i, "characteristics_ch1"] <- raw
        }
      }
    } else {
      meta[[key]] <- values
    }
  }
  meta
}

read_series_matrix_text <- function(path) {
  lines <- readLines(path, warn = FALSE)
  begin <- which(startsWith(lines, "!series_matrix_table_begin"))
  end <- which(startsWith(lines, "!series_matrix_table_end"))
  if (length(begin) == 0 || length(end) == 0) {
    stop("未找到 series_matrix_table 区块")
  }
  begin <- begin[1]
  end <- end[1]
  header <- strsplit(lines[begin + 1], "\t", fixed = TRUE)[[1]]
  sample_ids <- gsub("\"", "", header[-1])
  expr <- read.delim(path, sep = "\t", header = TRUE, skip = begin, nrows = end - begin - 2, check.names = FALSE, quote = "", comment.char = "", fill = TRUE, stringsAsFactors = FALSE)
  rownames(expr) <- gsub("\"", "", expr[[1]])
  expr[[1]] <- NULL
  colnames(expr) <- sample_ids
  expr <- as.matrix(expr)
  meta <- parse_geo_metadata(lines[1:begin], sample_ids)
  list(expr = expr, fdata = data.frame(row.names = rownames(expr)), pdata = meta)
}

read_series_matrix <- function(path) {
  if (requireNamespace("GEOquery", quietly = TRUE)) {
    gse <- getGEO(filename = path, GSEMatrix = TRUE)
    eset <- gse[[1]]
    expr <- exprs(eset)
    fdata <- fData(eset)
    pdata <- pData(eset)
    return(list(expr = expr, fdata = fdata, pdata = pdata))
  }
  read_series_matrix_text(path)
}

extract_metastasis <- function(df) {
  cols <- colnames(df)
  met_cols <- cols[grepl("metastasis|distant|m_stage|m stage|ajcc.*m|uicc.*m", cols, ignore.case = TRUE)]
  if (length(met_cols) == 0) {
    met_cols <- cols[grepl("stage|ajcc|pathologic.*stage|clinical.*stage", cols, ignore.case = TRUE)]
  }
  labels <- rep(NA, nrow(df))
  for (i in seq_len(nrow(df))) {
    row <- df[i, , drop = FALSE]
    val <- NA
    for (c in met_cols) {
      if (grepl("tstage|nstage|ptstage|pnstage", c, ignore.case = TRUE)) next
      v <- as.character(row[[c]])
      if (is.na(v) || v == "") next
      low <- tolower(v)
      if (grepl("m1|yes|positive|metastasis|distant", low)) {
        val <- 1
        break
      }
      if (grepl("m0|no|negative", low)) {
        val <- 0
        break
      }
      if (grepl("stage iv|stage 4|\\biv\\b", low)) {
        val <- 1
        break
      }
      stage_num <- suppressWarnings(as.numeric(gsub("[^0-9]", "", low)))
      if (!is.na(stage_num) && stage_num >= 4) {
        val <- 1
        break
      }
      if (!is.na(stage_num) && stage_num >= 1 && stage_num <= 3) {
        val <- 0
        next
      }
      if (grepl("stage i|stage ii|stage iii|\\bi\\b|\\bii\\b|\\biii\\b", low)) {
        val <- 0
      }
    }
    labels[i] <- val
  }
  labels
}

extract_survival <- function(df) {
  cols <- colnames(df)
  time_cols <- cols[grepl("overall_survival|os_time|os\\b|survival_time|days_to_death|days_to_last_follow_up", cols, ignore.case = TRUE)]
  status_cols <- cols[grepl("vital_status|status|death|event|os_status", cols, ignore.case = TRUE)]
  time <- if (length(time_cols) > 0) suppressWarnings(as.numeric(df[[time_cols[1]]])) else rep(NA, nrow(df))
  status_raw <- if (length(status_cols) > 0) as.character(df[[status_cols[1]]]) else rep(NA, nrow(df))
  status <- rep(NA, nrow(df))
  if (length(status_cols) > 0) {
    low <- tolower(status_raw)
    status[grepl("dead|deceased|1|yes", low)] <- 1
    status[grepl("alive|0|no", low)] <- 0
  }
  list(time = time, status = status)
}

assemble_dataset <- function(gse_id) {
  series_path <- file.path(geo_dir, paste0(gse_id, "_series_matrix.txt"))
  if (!file.exists(series_path)) {
    stop(paste("缺少文件", series_path))
  }
  obj <- read_series_matrix(series_path)
  expr <- map_probes_to_symbols(obj$expr, obj$fdata)
  pdata <- obj$pdata
  sample_id <- if ("geo_accession" %in% colnames(pdata)) pdata$geo_accession else rownames(pdata)
  clinical <- as.data.frame(pdata, stringsAsFactors = FALSE)
  sample_id <- gsub("\"", "", as.character(sample_id))
  colnames(expr) <- gsub("\"", "", colnames(expr))
  clinical$sample_id <- sample_id
  if (gse_id == "GSE15459") {
    outcome_path <- file.path(geo_dir, "GSE15459_outcome.xls")
    if (file.exists(outcome_path) && requireNamespace("readxl", quietly = TRUE)) {
      outc <- readxl::read_excel(outcome_path)
      outc_cols <- colnames(outc)
      sample_col <- outc_cols[grepl("gsm|sample", outc_cols, ignore.case = TRUE)]
      if (length(sample_col) > 0) {
        outc$sample_id <- as.character(outc[[sample_col[1]]])
        clinical <- merge(clinical, outc, by = "sample_id", all.x = TRUE)
      }
    }
  }
  if (gse_id == "GSE26901") {
    clinical_path <- file.path(geo_dir, "GSE26901_GC_KosinUniv_ClinicalInformation.txt")
    if (file.exists(clinical_path)) {
      outc <- read.delim(clinical_path, sep = "\t", header = TRUE, check.names = FALSE)
      outc_cols <- colnames(outc)
      sample_col <- outc_cols[grepl("gsm|sample", outc_cols, ignore.case = TRUE)]
      if (length(sample_col) > 0) {
        outc$sample_id <- as.character(outc[[sample_col[1]]])
        clinical <- merge(clinical, outc, by = "sample_id", all.x = TRUE)
      }
    }
  }
  clinical$metastasis <- extract_metastasis(clinical)
  surv <- extract_survival(clinical)
  clinical$survival_time <- surv$time
  clinical$survival_status <- surv$status
  expr <- as.data.frame(expr)
  expr$probe_id <- rownames(expr)
  expr <- expr[!duplicated(expr$probe_id), ]
  rownames(expr) <- expr$probe_id
  expr$probe_id <- NULL
  expr <- as.matrix(expr)
  expr <- suppressWarnings(matrix(as.numeric(expr), nrow = nrow(expr), dimnames = dimnames(expr)))
  list(expr = expr, clinical = clinical)
}

filter_samples <- function(expr, clinical) {
  clinical <- as.data.frame(clinical, stringsAsFactors = FALSE)
  clinical$sample_id <- gsub("\"", "", as.character(clinical$sample_id))
  colnames(expr) <- gsub("\"", "", colnames(expr))
  keep <- !is.na(clinical$metastasis)
  if (all(is.na(clinical$survival_time)) && all(is.na(clinical$survival_status))) {
    keep <- keep
  } else {
    keep <- keep & !is.na(clinical$survival_time) & !is.na(clinical$survival_status)
  }
  if (sum(keep) == 0) {
    keep <- !is.na(clinical$metastasis)
  }
  clinical <- clinical[keep, ]
  shared <- intersect(colnames(expr), clinical$sample_id)
  clinical <- clinical[clinical$sample_id %in% shared, , drop = FALSE]
  expr <- expr[, shared, drop = FALSE]
  list(expr = expr, clinical = clinical)
}

build_expression_matrix <- function(datasets) {
  gene_lists <- lapply(datasets, function(d) rownames(d$expr))
  all_genes <- Reduce(union, gene_lists)
  merged <- NULL
  meta <- NULL
  for (i in seq_along(datasets)) {
    d <- datasets[[i]]
    expr <- matrix(NA_real_, nrow = length(all_genes), ncol = ncol(d$expr), dimnames = list(all_genes, d$clinical$sample_id))
    expr[rownames(d$expr), ] <- d$expr
    expr <- suppressWarnings(apply(expr, 2, as.numeric))
    expr <- matrix(expr, nrow = length(all_genes), dimnames = list(all_genes, d$clinical$sample_id))
    if (is.null(merged)) {
      merged <- expr
    } else {
      merged <- cbind(merged, expr)
    }
    meta <- rbind(meta, data.frame(
      sample_id = d$clinical$sample_id,
      dataset = d$clinical$dataset,
      metastasis = d$clinical$metastasis,
      survival_time = d$clinical$survival_time,
      survival_status = d$clinical$survival_status
    ))
  }
  list(expr = merged, meta = meta)
}

impute_expression <- function(expr) {
  row_means <- rowMeans(expr, na.rm = TRUE)
  row_means[is.na(row_means)] <- 0
  idx <- which(is.na(expr), arr.ind = TRUE)
  if (nrow(idx) > 0) {
    expr[idx] <- row_means[idx[, 1]]
  }
  expr
}

combat_adjust <- function(expr, meta) {
  batch <- meta$dataset
  if (length(unique(batch)) < 2) {
    return(expr)
  }
  tryCatch({
    valid_levels <- length(unique(na.omit(meta$metastasis)))
    if (valid_levels < 2) {
      mod <- NULL
    } else {
      meta$metastasis_num <- suppressWarnings(as.numeric(as.character(meta$metastasis)))
      mod <- model.matrix(~ metastasis_num, data = meta)
    }
    ComBat(dat = expr, batch = batch, mod = mod, par.prior = TRUE, prior.plots = FALSE)
  }, error = function(e) {
    ComBat(dat = expr, batch = batch, mod = NULL, par.prior = TRUE, prior.plots = FALSE)
  })
}

filter_genes <- function(expr) {
  zero_rate <- rowMeans(expr <= 0 | is.na(expr))
  expr <- expr[zero_rate < 0.5, , drop = FALSE]
  vars <- apply(expr, 1, var, na.rm = TRUE)
  cutoff <- quantile(vars, 0.75, na.rm = TRUE)
  expr[vars >= cutoff, , drop = FALSE]
}

deg_by_dataset <- function(datasets) {
  deg_lists <- list()
  for (d in datasets) {
    expr <- d$expr
    labels <- d$clinical$metastasis
    keep <- !is.na(labels)
    expr <- expr[, keep, drop = FALSE]
    labels <- labels[keep]
    if (length(unique(labels)) < 2) {
      next
    }
    design <- model.matrix(~ 0 + factor(labels, levels = c(0, 1)))
    colnames(design) <- c("M0", "M1")
    fit <- lmFit(expr, design)
    contrast <- makeContrasts(M1 - M0, levels = design)
    fit2 <- contrasts.fit(fit, contrast)
    fit2 <- eBayes(fit2)
    tbl <- topTable(fit2, number = Inf, sort.by = "P")
    genes <- rownames(tbl)[abs(tbl$logFC) > 0.5 & tbl$P.Value < 0.05]
    deg_lists[[d$clinical$dataset[1]]] <- genes
  }
  deg_lists
}

run_wgcna <- function(expr, meta) {
  tryCatch({
    if ("allowWGCNAThreads" %in% getNamespaceExports("WGCNA")) {
      WGCNA::allowWGCNAThreads(nThreads = 1)
    } else {
      WGCNA::disableWGCNAThreads()
    }
    datExpr <- t(expr)
    gsg <- goodSamplesGenes(datExpr, verbose = 0)
    datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes, drop = FALSE]
    if (ncol(datExpr) < 30 || nrow(datExpr) < 10) {
      return(list(genes = rownames(expr), moduleColors = rep("grey", nrow(expr)), MEs = NULL))
    }
    if (ncol(datExpr) > 5000) {
      vars <- apply(datExpr, 2, var, na.rm = TRUE)
      top <- order(vars, decreasing = TRUE)[seq_len(5000)]
      datExpr <- datExpr[, top, drop = FALSE]
    }
    powers <- 1:12
    sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 0, corFnc = "cor")
    fit <- sft$fitIndices
    beta <- fit$Power[which(fit$SFT.R.sq > 0.9)][1]
    if (is.na(beta)) beta <- 6
    net <- blockwiseModules(
      datExpr,
      power = beta,
      TOMType = "unsigned",
      minModuleSize = 30,
      mergeCutHeight = 0.5,
      numericLabels = TRUE,
      saveTOMs = FALSE,
      maxBlockSize = 2000,
      nThreads = 1,
      verbose = 0
    )
    moduleLabels <- net$colors
    moduleColors <- labels2colors(moduleLabels)
    MEs <- moduleEigengenes(datExpr, colors = moduleColors)$eigengenes
    trait <- meta$metastasis[match(rownames(datExpr), meta$sample_id)]
    trait <- as.numeric(trait)
    cor_vals <- suppressWarnings(cor(MEs, trait, use = "p"))
    p_vals <- corPvalueStudent(cor_vals, nrow(datExpr))
    modules <- colnames(MEs)[abs(cor_vals) > 0.5 & p_vals < 0.05]
    genes <- names(moduleColors)[moduleColors %in% gsub("^ME", "", modules)]
    list(genes = genes, moduleColors = moduleColors, MEs = MEs)
  }, error = function(e) {
    list(genes = rownames(expr), moduleColors = rep("grey", nrow(expr)), MEs = NULL)
  })
}

run_lasso <- function(expr, labels, genes) {
  if (length(genes) < 2) {
    return(genes)
  }
  if (length(unique(na.omit(labels))) < 2) {
    return(genes)
  }
  X <- t(expr[genes, , drop = FALSE])
  if (ncol(X) < 2) {
    return(genes)
  }
  X <- scale(X)
  y <- labels
  cvfit <- cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 5)
  coef_mat <- coef(cvfit, s = "lambda.min")
  selected <- rownames(coef_mat)[coef_mat[, 1] != 0]
  selected <- selected[selected != "(Intercept)"]
  if (length(selected) == 0) {
    selected <- genes
  }
  selected
}

run_enrichment <- function(genes, out_prefix) {
  if (!requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
    return(NULL)
  }
  mapped <- bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  if (nrow(mapped) == 0) {
    return(NULL)
  }
  entrez <- unique(mapped$ENTREZID)
  go <- enrichGO(entrez, OrgDb = org.Hs.eg.db, ont = "BP", pvalueCutoff = 0.3, readable = TRUE)
  kegg <- enrichKEGG(entrez, organism = "hsa", pvalueCutoff = 0.15)
  write.csv(as.data.frame(go), paste0(out_prefix, "_go.csv"), row.names = FALSE)
  write.csv(as.data.frame(kegg), paste0(out_prefix, "_kegg.csv"), row.names = FALSE)
}

train_models <- function(expr, labels, out_dir) {
  X <- as.data.frame(t(expr))
  y <- factor(ifelse(labels == 1, "M1", "M0"), levels = c("M0", "M1"))
  if (nrow(X) < 4 || length(unique(y)) < 2) {
    results <- data.frame(model = character(), accuracy = numeric(), f1 = numeric(), auc = numeric())
    write.csv(results, file.path(out_dir, "model_performance.csv"), row.names = FALSE)
    return(list(results = results, best = NA_character_))
  }
  set.seed(42)
  idx <- sample(seq_len(nrow(X)), size = max(2, floor(0.8 * nrow(X))))
  train_X <- X[idx, , drop = FALSE]
  test_X <- X[-idx, , drop = FALSE]
  train_y <- y[idx]
  test_y <- y[-idx]
  lr_model <- NULL
  prob <- NULL
  fit_ok <- FALSE
  try({
    df <- train_X
    df$label <- train_y
    lr_model <- glm(label ~ ., data = df, family = binomial())
    prob <- as.numeric(predict(lr_model, newdata = test_X, type = "response"))
    fit_ok <- TRUE
  }, silent = TRUE)
  if (!fit_ok) {
    y_bin <- as.integer(train_y == "M1")
    cvfit <- cv.glmnet(as.matrix(train_X), y_bin, family = "binomial", alpha = 1, nfolds = 5)
    lr_model <- cvfit
    prob <- as.numeric(predict(cvfit, newx = as.matrix(test_X), s = "lambda.min", type = "response"))
  }
  pred <- factor(ifelse(prob >= 0.5, "M1", "M0"), levels = c("M0", "M1"))
  acc <- mean(pred == test_y)
  f1 <- F_meas(pred, test_y, positive = "M1")
  auc_val <- NA_real_
  roc_obj <- tryCatch(roc(response = test_y, predictor = prob, levels = c("M0", "M1"), direction = "<"), error = function(e) NULL)
  if (!is.null(roc_obj)) {
    auc_val <- as.numeric(auc(roc_obj))
  }
  results <- data.frame(model = "LR", accuracy = acc, f1 = f1, auc = auc_val)
  saveRDS(lr_model, file.path(out_dir, "model_LR.rds"))
  write.csv(results, file.path(out_dir, "model_performance.csv"), row.names = FALSE)
  list(results = results, best = "LR")
}

run_pipeline <- function() {
  stad_clinical_path <- file.path(stad_dir, "clinical.tsv")
  stad_meta <- NULL
  if (file.exists(stad_clinical_path)) {
    stad_meta <- read.delim(stad_clinical_path, sep = "\t", header = TRUE, check.names = FALSE)
    stad_meta$metastasis <- extract_metastasis(stad_meta)
    surv <- extract_survival(stad_meta)
    stad_meta$survival_time <- surv$time
    stad_meta$survival_status <- surv$status
    write.csv(stad_meta, file.path(processed_dir, "stad_clinical_processed.csv"), row.names = FALSE)
  }

  datasets <- list()
  for (gse_id in gse_ids) {
    obj <- assemble_dataset(gse_id)
    obj$clinical$dataset <- gse_id
    obj <- filter_samples(obj$expr, obj$clinical)
    if (ncol(obj$expr) == 0 || nrow(obj$clinical) == 0) {
      next
    }
    datasets[[gse_id]] <- obj
  }
  if (length(datasets) == 0) {
    stop("没有可用的带有转移标签的样本")
  }

  integrated <- build_expression_matrix(datasets)
  expr <- integrated$expr
  meta <- integrated$meta
  labels <- meta$metastasis
  names(labels) <- meta$sample_id
  keep_samples <- !is.na(labels)
  expr <- expr[, keep_samples, drop = FALSE]
  meta <- meta[meta$sample_id %in% colnames(expr), , drop = FALSE]
  labels <- labels[keep_samples]

  expr <- impute_expression(expr)
  expr_combat <- combat_adjust(expr, meta)
  expr_filtered <- filter_genes(expr_combat)

  deg_lists <- deg_by_dataset(datasets)
  deg_mode <- "intersection"
  deg_genes <- if (length(deg_lists) == 0) character(0) else if (deg_mode == "intersection") Reduce(intersect, deg_lists) else unique(unlist(deg_lists))
  write.csv(deg_lists, file.path(processed_dir, "deg_genes_by_dataset.csv"))
  write.csv(deg_genes, file.path(processed_dir, "deg_genes_union_or_intersection.csv"), row.names = FALSE)

  labels <- labels[colnames(expr_filtered)]
  if (nrow(expr_filtered) == 0 || ncol(expr_filtered) == 0) {
    write.csv(character(0), file.path(processed_dir, "wgcna_core_genes.csv"), row.names = FALSE)
    write.csv(character(0), file.path(processed_dir, "core_genes.csv"), row.names = FALSE)
    write.csv(character(0), file.path(processed_dir, "biomarkers.csv"), row.names = FALSE)
    write.csv(data.frame(), file.path(processed_dir, "biomarker_expression.csv"), row.names = FALSE)
    model_out <- train_models(expr_filtered, labels, output_dir)
    write.csv(meta, file.path(processed_dir, "integrated_metadata.csv"), row.names = FALSE)
    write.csv(t(expr_filtered), file.path(processed_dir, "integrated_expression.csv"))
  } else {
    wgcna_res <- run_wgcna(expr_filtered, meta)
    write.csv(wgcna_res$genes, file.path(processed_dir, "wgcna_core_genes.csv"), row.names = FALSE)

    core_genes <- intersect(deg_genes, wgcna_res$genes)
    if (length(core_genes) == 0) {
      core_genes <- wgcna_res$genes
    }
    write.csv(core_genes, file.path(processed_dir, "core_genes.csv"), row.names = FALSE)

    biomarkers <- run_lasso(expr_filtered, labels, core_genes)
    biomarkers <- intersect(biomarkers, rownames(expr_filtered))
    if (length(biomarkers) == 0) {
      biomarkers <- rownames(expr_filtered)
    }
    write.csv(biomarkers, file.path(processed_dir, "biomarkers.csv"), row.names = FALSE)

    expr_biomarkers <- expr_filtered[biomarkers, , drop = FALSE]
    write.csv(t(expr_biomarkers), file.path(processed_dir, "biomarker_expression.csv"))

    run_enrichment(biomarkers, file.path(processed_dir, "biomarkers_enrichment"))

    model_out <- train_models(expr_biomarkers, labels, output_dir)
    write.csv(meta, file.path(processed_dir, "integrated_metadata.csv"), row.names = FALSE)
    write.csv(t(expr_filtered), file.path(processed_dir, "integrated_expression.csv"))
  }
}

if (Sys.getenv("RUN_PIPELINE") != "0") {
  run_pipeline()
}
