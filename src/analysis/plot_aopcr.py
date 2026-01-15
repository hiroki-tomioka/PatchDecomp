import sys
import argparse
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("..")
from utils import make_folder, save_args, load_csv, MODEL_COLORS


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True, help="path to the result folder to analyze"
)
parser.add_argument(
    "--ks",
    nargs="*",
    type=float,
    default=[5.0, 7.5, 10.0, 12.5, 15.0],
    help="list of removal ratio",
)
parser.add_argument(
    "--unique_ids",
    nargs="*",
    type=str,
    default=["BE"],
    help="unique ID",
)
args = parser.parse_args()

methods = ["PatchDecomp", "TFT-patch", "TFT-point", "random"]
x = args.ks

# Font
mpl.rcParams["font.family"] = "Times New Roman"

# Output folder
output_path = make_folder(args.path + "AOPCR/")

# Save command-line parameters
save_args(args.__dict__, output_path)


for unique_id in args.unique_ids:
    anlz_paths = {method: [] for method in methods}
    comps = {method: [] for method in methods}

    # Extract analysis results to handle
    expt_paths_PatchDecomp = sorted(
        glob.glob(args.path + "seed*/results/PatchDecomp/analysis/AOPCR/")
    )
    expt_paths_TFT = sorted(glob.glob(args.path + "seed*/results/TFT/analysis/AOPCR/"))
    for expt_path in expt_paths_PatchDecomp + expt_paths_TFT:
        for method in methods:
            paths = sorted(glob.glob(expt_path + f"{method}/*/"))
            if len(paths) > 0:
                anlz_paths[method].append(paths[-1])

    # Collect "comperehensiveness" results
    for method in methods:
        if "TFT" in method:
            model = "TFT"
        else:
            model = "PatchDecomp"
        for path in anlz_paths[method]:
            metrics = load_csv(f"{path}/metrics.csv")
            comps[method].append(metrics["comprehensiveness"].dropna(how="all"))

    # Plot line graph
    fig = plt.figure(figsize=(4.5, 1.7))
    for i, method in enumerate(methods):
        if "PatchDecomp" in method:
            model = "PatchDecomp"
            color = MODEL_COLORS[model]
            marker = "o"
            linestyle = "-"
            label = "PatchDecomp"
        elif "random" in method:
            model = "PatchDecomp"
            color = "black"
            marker = "D"
            linestyle = "-"
            label = method
        elif "TFT" in method:
            model = "TFT"
            if "TFT-point" in method:
                color = MODEL_COLORS[model]
                marker = "v"
                linestyle = "--"
                label = method
            elif "TFT-patch" in method:
                color = MODEL_COLORS[model]
                marker = "^"
                linestyle = "-"
                label = method
        comps[method] = pd.concat(comps[method], axis=1)
        comps[method]["mean"] = comps[method].mean(axis=1)
        comps[method]["std"] = comps[method].std(axis=1)
        print("--")
        print(method, "comprehensiveness")
        print(comps[method])
        plt.errorbar(
            x,
            comps[method].loc[comps[method].index != "AOPCR", "mean"],
            yerr=comps[method].loc[comps[method].index != "AOPCR", "std"],
            capsize=5,
            marker=marker,
            markersize=6,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    # plt.title(f'{metric} ({unique_id})')
    plt.ylim(0, 12)
    plt.xlabel("k [%]")
    plt.ylabel(f"AOPCR ({unique_id})")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/comprehensiveness.svg")
    plt.savefig(f"{output_path}/comprehensiveness.png")
