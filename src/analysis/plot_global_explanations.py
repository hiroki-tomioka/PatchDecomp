import os
import sys
import argparse
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast, core

sys.path.append("..")
from models.patchdecomp import PatchDecomp
from utils import make_folder, save_args


def plot_patch_importance(model, features, patch_importance, path):
    fig = plt.figure(figsize=(3.5, 3))
    for row, feature_name in enumerate(features):
        ax = fig.add_subplot(len(features), 1, row + 1)
        xticks = []
        for patch in range(model.n_patches):
            name = f"{feature_name}_{patch}"
            if name in model.patch_names:
                x0 = model.patch_begins[patch]
                xticks.append(x0)
                x1 = model.patch_ends[patch]
                ax.axvspan(
                    x0, x1, alpha=min(1, patch_importance[name]), color="tab:red"
                )
        xticks.append(x1)
        ax.set_xlim(model.t[0] - 5, model.t[-1] + 5)
        ax.set_title(feature_name)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xticks(xticks)
    plt.xticks(rotation=45)
    # plt.suptitle("Global patch importance")
    plt.tight_layout(h_pad=0.5)
    plt.savefig(path)


def plot_point_importance(model, features, point_importance, path):
    fig = plt.figure(figsize=(3.5, 3))
    for row, feature_name in enumerate(features):
        ax = fig.add_subplot(len(features), 1, row + 1)
        importances = point_importance[feature_name]
        for i, importance in enumerate(importances):
            step = model.t[i]
            if not np.isnan(importance):
                ax.axvspan(step, step + 1, alpha=min(1, importance), color="tab:red")
        xticks = []
        for patch in range(model.n_patches):
            name = f"{feature_name}_{patch}"
            if name in model.patch_names:
                x0 = model.patch_begins[patch]
                xticks.append(x0)
                x1 = model.patch_ends[patch]
        xticks.append(x1)
        ax.set_xlim(model.t[0] - 5, model.t[-1] + 5)
        ax.set_title(feature_name)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xticks(xticks)
    plt.xticks(rotation=45)
    # plt.suptitle("Global point importance")
    plt.tight_layout(h_pad=0.5)
    plt.savefig(path)


def plot_global_explanations(
    features, model, name1, name2, importance1, importance2, path
):
    fig = plt.figure(figsize=(5.5, 3.2))
    for row, feature_name in enumerate(features):
        ax1 = fig.add_subplot(len(features), 2, 2 * row + 1)
        ax2 = fig.add_subplot(len(features), 2, 2 * row + 2)
        if row == 0:
            ax1.text(0.5, 1.3, name1, ha="center", va="center", transform=ax1.transAxes)
            ax2.text(0.5, 1.3, name2, ha="center", va="center", transform=ax2.transAxes)
        xticks = []
        for patch in range(model.n_patches):
            name = f"{feature_name}_{patch}"
            if name in model.patch_names:
                x0 = model.patch_begins[patch]
                xticks.append(x0)
                x1 = model.patch_ends[patch]
                ax1.axvspan(x0, x1, alpha=min(1, importance1[name]), color="tab:red")
        point_importances = importance2[feature_name]
        for i, importance in enumerate(point_importances):
            step = model.t[i]
            if not np.isnan(importance):
                ax2.axvspan(step, step + 1, alpha=min(1, importance), color="tab:red")
        xticks.append(x1)
        ax1.set_xlim(model.t[0] - 5, model.t[-1] + 5)
        ax2.set_xlim(model.t[0] - 5, model.t[-1] + 5)
        ax1.text(
            -0.25, 0.5, feature_name, ha="center", va="center", transform=ax1.transAxes
        )
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    fig.autofmt_xdate(rotation=45, ha="center")
    # plt.suptitle("Global patch importance")
    plt.tight_layout(h_pad=0.5, w_pad=0.3)
    plt.savefig(path)


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path_patchdecomp", type=str, required=True)
parser.add_argument("--path_tft", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# Font
mpl.rcParams["font.family"] = "Times New Roman"

core.MODEL_FILENAME_DICT["autopatchdecomp"] = PatchDecomp

methods = ["PatchDecomp", "TFT-patch", "TFT-point"]
paths = {method: None for method in methods}
importances = {method: None for method in methods}
color_scaler = 10


# Extract analysis results to handle
paths["PatchDecomp"] = sorted(
    glob.glob(args.path_patchdecomp + "analysis/AOPCR/PatchDecomp/*/")
)[-1]
paths["TFT-point"] = sorted(glob.glob(args.path_tft + "analysis/AOPCR/TFT-point/*/"))[
    -1
]

# Output folder
output_path = make_folder(paths["PatchDecomp"] + "../../../global_explanation/")

# Save command-line parameters
save_args(args.__dict__, output_path)

# Load model
nf = NeuralForecast.load(paths["PatchDecomp"] + "../../../../", weights_only=False)
model = nf.models[0]
features = ["y"] + model.hist_exog_list + model.futr_exog_list

# Load PatchDecomp's analysis results
decomp = np.load(f"{paths['PatchDecomp']}/y_decomps.npy")
decomp = np.sum(abs(decomp), axis=(0, 1, 3, 4))
decomp /= np.sum(decomp) * color_scaler
decomp *= decomp.shape

# Patch importance of PatchDecomp
importances["PatchDecomp"] = {
    key: value for key, value in zip(model.patch_names, decomp)
}
# Plot patch importance of PatchDecomp
plot_patch_importance(
    model,
    features,
    importances["PatchDecomp"],
    output_path + "/PatchDecomp.svg",
)

# Load TFT-point's analysis results
feature_importances = np.load(f"{paths['TFT-point']}/feature_importances.npy")
feature_importances = np.sum(abs(feature_importances), axis=(0, 1))
feature_importances /= np.nansum(feature_importances) * color_scaler
feature_importances *= np.count_nonzero(~np.isnan(feature_importances))

# Point-wise importance of TFT
importances["TFT-point"] = {
    key: value for key, value in zip(features, feature_importances)
}
# Plot point-wise importance of TFT
plot_point_importance(
    model,
    features,
    importances["TFT-point"],
    output_path + "/TFT_point.svg",
)

# Patch-wise importance of TFT
importances["TFT-patch"] = {}
for feature in features:
    for i in range(len(importances["TFT-point"][feature]) // model.patch_size):
        tag = f"{feature}_{i}"
        importances["TFT-patch"][tag] = (
            importances["TFT-point"][feature][
                (model.patch_size * i) : (model.patch_size * (i + 1))
            ]
            .sum()
            .item()
        )
scaler = np.nansum(np.array(list(importances["TFT-patch"].values()))) * color_scaler
importances["TFT-patch"] = {
    key: value / scaler * len(importances["TFT-patch"])
    for key, value in importances["TFT-patch"].items()
    if not np.isnan(value)
}
# Plot patch-wise importance of TFT
plot_patch_importance(
    model,
    features,
    importances["TFT-patch"],
    output_path + "/TFT_patch.svg",
)

# Plot importances of PatchDecomp and TFT
plot_global_explanations(
    features,
    model,
    "PatchDecomp",
    "TFT",
    importances["PatchDecomp"],
    importances["TFT-point"],
    output_path + "/PatchDecomp_TFT.pdf",
)

print("\nThe results were saved in", output_path)
