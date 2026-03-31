from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse


def find_model_performance_files(base_dir: Path):
    return list(base_dir.rglob('**/model_performance.csv'))


def load_and_annotate(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # annotate
    df = df.copy()
    df['source_path'] = str(path)
    df['run_dir'] = str(path.parent)
    # try to extract a timestamp-like name
    df['run_name'] = path.parent.name
    return df


def aggregate(base_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_model_performance_files(base_dir)
    rows = []
    for p in files:
        df = load_and_annotate(p)
        if df is None:
            continue
        rows.append(df)
    if not rows:
        print('No model_performance.csv files found under', base_dir)
        return
    all_df = pd.concat(rows, ignore_index=True, sort=False)
    agg_csv = out_dir / 'aggregated_model_performance.csv'
    all_df.to_csv(agg_csv, index=False)
    # compute summary stats per model
    numeric_cols = [c for c in ['accuracy', 'f1', 'auc'] if c in all_df.columns]
    summary = all_df.groupby('model')[numeric_cols].agg(['count', 'mean', 'std']).round(4)
    summary_csv = out_dir / 'model_performance_summary.csv'
    summary.to_csv(summary_csv)
    # save JSON summary
    summary_json = out_dir / 'model_performance_summary.json'
    summary.to_json(summary_json)
    print('Aggregated', len(files), 'files ->', agg_csv)
    print('Summary saved to', summary_csv)
    # plot mean accuracy with error bars if accuracy exists
    if 'accuracy' in all_df.columns:
        grp = all_df.groupby('model')['accuracy'].agg(['mean', 'std']).reindex()
        plt.figure(figsize=(8, 4))
        models = grp.index.tolist()
        means = grp['mean'].fillna(0).values
        errs = grp['std'].fillna(0).values
        plt.bar(models, means, yerr=errs, capsize=4)
        plt.ylabel('Accuracy')
        plt.title('Model accuracy mean ± std')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_path = out_dir / 'model_accuracy_mean_std.png'
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print('Plot saved to', fig_path)


def main():
    parser = argparse.ArgumentParser(description='Collect model_performance.csv files and aggregate statistics')
    parser.add_argument('--workspace', '-w', default='.', help='workspace root (default: current dir)')
    parser.add_argument('--out', '-o', default='output/summary', help='output folder for aggregated results')
    args = parser.parse_args()
    base = Path(args.workspace).resolve()
    out_dir = base / args.out
    aggregate(base, out_dir)


if __name__ == '__main__':
    main()
