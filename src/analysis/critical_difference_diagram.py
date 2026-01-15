import sys
import argparse
import glob
import yaml
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("..")
from utils import make_folder, save_args, MODEL_COLORS


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True, help="path to the result folder to analyze"
)
args = parser.parse_args()

# Font
mpl.rcParams["font.family"] = "Times New Roman"

# Output folder
output_path = make_folder(args.path + "critical_difference_diagram/")

# Save command-line parameters
save_args(args.__dict__, output_path)


# Collect the names of folder, dataset, and model
task = args.path.split("/")[-2]
if task == "ltsf":
    figsize = (9, 2.8)
    result_folders = sorted(glob.glob(args.path + "seed*/"))
    result_folders = [folder for folder in result_folders if "critical" not in folder]
    datasets = sorted(glob.glob(result_folders[0] + "results/*/"))
    models = sorted(glob.glob(datasets[0] + "*/"))
    datasets = list(map(lambda path: path.split("/")[-2], datasets))
    models = list(map(lambda path: path.split("/")[-2], models))
elif task == "epf":
    figsize = (9, 2)
    result_folders = sorted(glob.glob(args.path + "*/seed*/"))
    result_folders = [folder for folder in result_folders if "critical" not in folder]
    datasets = []
    for folder in result_folders:
        dataset = folder.split("/")[-3]
        if dataset not in datasets:
            datasets.append(dataset)
    models = sorted(glob.glob(result_folders[0] + "results/*/"))
    models = list(map(lambda path: path.split("/")[-2], models))

# Aggregate metrics
metrics = []
for folder in result_folders:
    for dataset in datasets:
        for model in models:
            if task == "ltsf":
                file_path = f"{folder}/results/{dataset}/{model}/metric.yml"
            elif task == "epf":
                if dataset != folder.split("/")[-3]:
                    continue
                file_path = f"{folder}/results/{model}/metric.yml"
            with open(file_path, "r") as f:
                metric = yaml.safe_load(f)
                metric["experiment"] = dataset
                metrics += [metric]
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(output_path + "metrics.csv", index=False)
print(df_metrics)
print(df_metrics["experiment"])

# Take the average of the metrics
df_metrics_mean = df_metrics.groupby(["experiment", "model"])[
    ["MAE", "MSE", "time"]
].mean()
df_metrics_mean.to_csv(output_path + "metrics_mean.csv")
print(df_metrics_mean)

# Make critical difference diagrams
ps, dfs = [], []
# Critical difference diagrams for each dataset
experiments = df_metrics["experiment"].unique()
for expt in experiments:
    df = df_metrics[df_metrics["experiment"] == expt]
    df = df.assign(MAE_rank=df["MAE"].rank(pct=True, ascending=False))
    df = df.assign(MSE_rank=df["MSE"].rank(pct=True, ascending=False))
    dfs.append(df)
    for error in ["MAE", "MSE"]:
        data = [
            df.loc[ids, f"{error}_rank"].values
            for ids in df.groupby("model").groups.values()
        ]
        H, p = ss.kruskal(*data)
        print(expt, error, len(data), p)
        ps.append([expt, error, p])
        test_results = sp.posthoc_conover(
            df, val_col=f"{error}_rank", group_col="model", p_adjust="holm"
        )
        avg_rank = df[error].rank(pct=True, ascending=False).groupby(df["model"]).mean()
        plt.figure(figsize=figsize)
        plt.rcParams["font.size"] = 15
        sp.critical_difference_diagram(
            avg_rank, test_results, color_palette=MODEL_COLORS
        )
        plt.tight_layout()
        plt.savefig(output_path + f"{expt}_{error}.svg")
        plt.close()
# Critical difference diagrams for all datasets
df_all = pd.concat(dfs, axis=0)
for error in ["MAE", "MSE"]:
    data = [
        df_all.loc[ids, f"{error}_rank"].values
        for ids in df_all.groupby("model").groups.values()
    ]
    H, p = ss.kruskal(*data)
    print("all", error, len(data), p)
    ps.append(["all", error, p])
    test_results = sp.posthoc_conover(
        df_all, val_col=f"{error}_rank", group_col="model", p_adjust="holm"
    )
    avg_rank = (
        df_all[error].rank(pct=True, ascending=False).groupby(df_all["model"]).mean()
    )
    plt.figure(figsize=figsize)
    plt.rcParams["font.size"] = 15
    sp.critical_difference_diagram(avg_rank, test_results, color_palette=MODEL_COLORS)
    plt.tight_layout()
    plt.savefig(output_path + f"all_{error}.svg")
    plt.close()

# Save p-values
ps = pd.DataFrame(ps, columns=["experiment", "metric", "p"])
ps.to_csv(output_path + "ps.csv", index=False)

print("\nThe results were saved in", output_path)
