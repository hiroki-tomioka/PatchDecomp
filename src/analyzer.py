import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.losses.numpy import mae, mse
import plotly.express as px

from utils import suppress_output_except_tqdm
from gui import VisualizerPatchDecomp


class AOPCRCalculator(VisualizerPatchDecomp):
    def __init__(self, method: str, dirname: str, seed: int):
        super().__init__(dirname)
        self.method = method
        self.names = ["y"] + self.model.hist_exog_list + self.model.futr_exog_list
        np.random.seed(seed)

    def calculate_aopcr(self, test_size, ks, replace, path):
        self.test_size = test_size
        self.ks = ks
        self.replace = replace
        self.path = path
        features = self.feature_names()
        ys_ans, ys, all_y_decomps, all_feature_importances = [], [], [], []
        ys_removed = {f"k{k}": [] for k in self.ks}

        with suppress_output_except_tqdm() as tqdm_out_stream:
            # Calculate by unique ID
            for unique_id in self.unique_ids:
                self.y_decomps_unique, self.feature_importance_unique = [], []
                self.df_unique = self.df[self.df["unique_id"] == unique_id].reset_index(
                    drop=True
                )
                self.data_length = self.df_unique.shape[0]
                with tqdm(
                    range(
                        self.data_length - self.test_size,
                        self.data_length - self.horizon,
                    ),
                    file=tqdm_out_stream,
                ) as pbar_steps:
                    # Calculate by step
                    for index in pbar_steps:
                        pbar_steps.set_description(f"{unique_id} ")

                        # Extract data to handle
                        begin = index - self.input_size + 1
                        end = index + self.horizon + 1
                        self.df_window = self.df_unique.iloc[begin:end].reset_index(
                            drop=True
                        )

                        # Predict w/o removal
                        y_ans, y, feature_importance, patch_importance = (
                            self.predict_wo_removal(features)
                        )
                        ys_ans.append(y_ans)
                        ys.append(y.squeeze())

                        # Predict w/ removal
                        self.df_window["step"] = self.t
                        for k in self.ks:
                            y_removed = self.predict_w_removal(
                                index,
                                unique_id,
                                k,
                                y,
                                feature_importance,
                                patch_importance,
                            )
                            ys_removed[f"k{k}"].append(y_removed.squeeze())

                if self.method == "PatchDecomp":
                    y_decomps = np.concatenate(self.y_decomps_unique)
                    all_y_decomps.append(y_decomps)
                elif "TFT" in self.method:
                    feature_importances = np.stack(self.feature_importance_unique)
                    all_feature_importances.append(feature_importances)

        # Save y_decomp
        if self.method == "PatchDecomp":
            all_y_decomps = np.stack(
                all_y_decomps
            )  # [N_id, N_pred, N_patch, H/L_patch, L_patch]
            np.save(self.path + "/y_decomps.npy", all_y_decomps)
        elif "TFT" in self.method:
            all_feature_importances = np.stack(all_feature_importances)
            np.save(path + "/feature_importances.npy", all_feature_importances)

        # Aggregate results
        self.aggregate_results(ys_ans, ys, ys_removed)

    def predict_w_removal(
        self, index, unique_id, k, y, feature_importance, patch_importance
    ):
        # Calculate important parts
        important_parts = self.calc_important_parts(
            feature_importance, patch_importance, k
        )

        # Remove top k% important parts
        df_window_removed = self.remove_values(important_parts)
        dataset_removed, *_ = TimeSeriesDataset.from_df(
            df_window_removed, static_df=self.static_df
        )
        y_removed = self.model.decompose(dataset=dataset_removed)

        # Plot removed time series
        if index == self.data_length - self.horizon - 1:
            self.plot_removed_timeseries(
                y,
                y_removed,
                df_window_removed,
                k,
                unique_id,
                index,
                important_parts,
                self.path + f"/removed_top{k}_{unique_id}_{index}.png",
            )

        return y_removed

    def aggregate_results(self, ys_ans, ys, ys_removed):
        maes, mses, comprehensivenesses = {}, {}, {}
        comprehensiveness = 0

        # Actual values
        ys_ans = np.concatenate(ys_ans)
        # Predicted values w/o removal (top 0% patches are removed)
        ys_removed["none"] = np.concatenate(ys)
        # Predicted values w/ removal
        for k in self.ks:
            ys_removed[f"k{k}"] = np.concatenate(ys_removed[f"k{k}"])
            maes[f"k{k}"] = float(mae(ys_ans, ys_removed[f"k{k}"]))
            mses[f"k{k}"] = float(mse(ys_ans, ys_removed[f"k{k}"]))
            comprehensivenesses[f"k{k}"] = sum(
                abs(ys_removed["none"] - ys_removed[f"k{k}"])
            ) / len(ys_removed["none"])
            comprehensiveness += comprehensivenesses[f"k{k}"]
        comprehensiveness /= len(self.ks)
        df_metrics = pd.DataFrame(
            {
                "MSE": mses,
                "MAE": maes,
                "comprehensiveness": comprehensivenesses,
            }
        )
        df_metrics.loc["AOPCR"] = [np.nan, np.nan, comprehensiveness]
        print(df_metrics)
        df_metrics.to_csv(self.path + "/metrics.csv")

    def plot_removed_timeseries(
        self,
        y,
        y_removed,
        df_window_removed,
        k,
        unique_id,
        index,
        removed_patches,
        path,
    ):
        fig = plt.figure(figsize=(10, 10))
        for row, feature_name in enumerate(self.names):
            ax = fig.add_subplot(len(self.names), 1, row + 1)
            if "point" not in self.method:
                for patch in range(self.n_patches):
                    name = f"{feature_name}_{patch}"
                    if name in self.patch_names:
                        x0 = self.patch_begins[patch]
                        x1 = self.patch_ends[patch]
                        if name in removed_patches:
                            color = "gray"
                        else:
                            color = "white"
                        ax.axvspan(x0, x1, alpha=0.3, color=color)
            ax.plot(self.t, df_window_removed[feature_name], c="black")

            ax.set_title(feature_name)
            if feature_name == "y":
                plt.plot(
                    self.t_futr,
                    y[0],
                    c="tab:red",
                    label="prediction (w/o removal)",
                )
                plt.plot(
                    self.t_futr,
                    y_removed[0],
                    c="tab:blue",
                    label="prediction (w/ removal)",
                )
                plt.legend(loc="upper left")
        plt.suptitle(
            f"Removed top {k}% patches ({unique_id}; {self.df_unique.loc[index, 'ds']})",
        )
        plt.tight_layout()
        plt.savefig(path)

    def feature_names(self):
        # Names of all features
        features = []
        for feature_name in self.names:
            for patch in range(self.n_patches):
                name = f"{feature_name}_{patch}"
                if name in self.patch_names:
                    features.append(name)
        return features

    def extract_important_patches(self, patch_importance, k):
        # Sort patches based on patch importance
        sorted_patch_importance = sorted(
            patch_importance.items(), key=lambda x: x[1], reverse=True
        )
        # Extract the top k% patches
        removed_patches_num = int(len(sorted_patch_importance) * k // 100)
        removed_patches = sorted_patch_importance[:removed_patches_num]
        removed_patches = [key for key, _ in removed_patches]
        return removed_patches

    def remove_values(self, removed_parts):
        # Remove important values
        df_window_removed = self.df_window.copy()
        if "point" in self.method:  # TFT-point
            for _, row in removed_parts.iterrows():
                idx = row["index"]
                col = row["column"]
                if col == "observed_target":
                    col = "y"
                df_window_removed = self.replace_values(df_window_removed, idx, col)
        else:  # PatchDecomp, random, TFT-patch
            for feature_name in self.names:
                for patch in range(self.n_patches):
                    name = f"{feature_name}_{patch}"
                    if name in removed_parts:
                        removed_var = name.split("_")[0]
                        if removed_var == "week":
                            removed_var = "week_day"
                        df_window_removed = self.replace_values(
                            df_window_removed, patch, removed_var
                        )
        df_window_removed = df_window_removed.drop("step", axis=1)
        return df_window_removed


class AOPCRCalculatorPatchDecomp(AOPCRCalculator):
    def __init__(self, method: str, dirname: str, seed: int):
        super().__init__(method, dirname, seed)

        self.t_hist = self.model.t_hist
        self.t_futr = self.model.t_futr
        self.t = self.model.t
        self.patch_begins = self.model.patch_begins
        self.patch_ends = self.model.patch_ends
        self.n_patches = self.model.n_patches
        self.patch_names = self.model.patch_names

    def predict_wo_removal(self, features):
        # Predict w/o removal
        y_ans = self.df_window["y"].tail(self.horizon)
        dataset, *_ = TimeSeriesDataset.from_df(
            self.df_window, static_df=self.static_df
        )
        y = self.model.decompose(dataset=dataset)

        # Calculate patch importance based on the designated method
        patch_importance = self.calculate_patch_importance(y, features)
        return y_ans, y, None, patch_importance

    def calc_important_parts(self, feature_importance, patch_importance, k):
        important_patches = self.extract_important_patches(patch_importance, k)
        return important_patches

    def calculate_patch_importance(self, feature_importance, features):
        if self.method == "PatchDecomp":
            y_decomp = self.model.y_decomp
            y_scale = (
                self.model.scalers["y"]
                .x_scale.reshape(1, 1, 1, 1)
                .to("cpu")
                .detach()
                .numpy()
            )
            y_decomp /= y_scale  # [N_pred, N_patch, H/L_patch, L_patch]
            self.y_decomps_unique.append(y_decomp)
            _patch_importance = abs(y_decomp).sum(axis=(0, 2, 3))
            _patch_importance /= (
                y_decomp.shape[0] + y_decomp.shape[2] + y_decomp.shape[3]
            )  # [N_patch]
            patch_importance = {
                key: float(value) for key, value in zip(features, _patch_importance)
            }
        elif self.method == "random":
            patch_importance = {key: np.random.random() for key in features}
        return patch_importance

    def replace_values(self, df, patch, removed_var):
        x0 = self.patch_begins[patch]
        x1 = self.patch_ends[patch]
        # Values before replacement
        removed_vals = self.df_window.loc[
            (self.df_window["step"] >= x0) & (self.df_window["step"] <= x1),
            removed_var,
        ]
        # Values after replacement
        ## "Local" focuses only on values in the patch
        ## "Global" uses all values in test dataset
        if self.replace == "zero":
            replacing_vals = 0
        elif self.replace == "max-minus":
            local_max = max(self.df_window[removed_var])
            replacing_vals = local_max - removed_vals
        elif self.replace == "swap":
            replacing_vals = removed_vals.iloc[::-1].values
        elif self.replace == "local-mean":
            local_mean = self.df_window[removed_var].mean()
            replacing_vals = local_mean
        elif self.replace == "global-mean":
            global_mean = self.df_unique[removed_var].tail(self.test_size).mean()
            replacing_vals = global_mean
        elif self.replace == "local-noise":
            local_mean = self.df_window[removed_var].mean()
            local_std = removed_vals.std()
            indices = self.df_window.index[
                (self.df_window["step"] >= x0) & (self.df_window["step"] <= x1)
            ].tolist()
            replacing_vals = pd.Series(
                np.random.normal(
                    loc=local_mean, scale=local_std, size=len(removed_vals)
                ),
                index=indices,
            )
        elif self.replace == "global-noise":
            global_mean = self.df_unique[removed_var].tail(self.test_size).mean()
            global_std = self.df_unique[removed_var].tail(self.test_size).std()
            indices = self.df_window.index[
                (self.df_window["step"] >= x0) & (self.df_window["step"] <= x1)
            ].tolist()
            replacing_vals = pd.Series(
                np.random.normal(
                    loc=global_mean,
                    scale=global_std,
                    size=len(removed_vals),
                ),
                index=indices,
            )
        # Replace the patch
        df.loc[
            (df["step"] >= x0) & (df["step"] <= x1),
            removed_var,
        ] = replacing_vals
        return df


class AOPCRCalculatorTFT(AOPCRCalculator):
    def __init__(self, method: str, dirname: str, seed: int):
        super().__init__(method, dirname, seed)

        self.t_hist = np.arange(self.model.input_size) - self.model.input_size + 1
        self.t_futr = np.arange(self.model.h) + 1
        self.t = np.concatenate([self.t_hist, self.t_futr])
        self._patchdecomp = NeuralForecast.load(
            f"{dirname}/../PatchDecomp/", weights_only=False
        ).models[0]
        self.patch_begins = self._patchdecomp.patch_begins
        self.patch_ends = self._patchdecomp.patch_ends
        self.n_patches = self._patchdecomp.n_patches
        self.patch_names = self._patchdecomp.patch_names

    def predict_wo_removal(self, features):
        # Predict w/o removal
        y_ans = self.df_window["y"].tail(self.horizon)
        dataset, *_ = TimeSeriesDataset.from_df(
            self.df_window, static_df=self.static_df
        )
        y = self.model.decompose(dataset=dataset)
        # Variable importances
        past_variable_importance = self.model.feature_importances()[
            "Past variable importance over time"
        ]
        future_variable_importance = self.model.feature_importances()[
            "Future variable importance over time"
        ]
        # Attention weights
        past_mean_attention = self.model.attention_weights()[
            self.model.input_size :, :
        ].mean(axis=0)[: self.model.input_size]
        future_mean_attention = self.model.attention_weights()[
            self.model.input_size :, :
        ].mean(axis=0)[self.model.input_size :]
        past_feature_importance = past_variable_importance.multiply(
            past_mean_attention, axis=0
        )
        future_feature_importance = future_variable_importance.multiply(
            future_mean_attention, axis=0
        )
        # Feature importance of variables by time
        feature_importance = pd.concat(
            [past_feature_importance, future_feature_importance], axis=0
        ).reset_index(drop=True)
        _feature_importance = feature_importance.copy().rename(
            columns={"observed_target": "y"}
        )
        _feature_importance = _feature_importance.reindex(columns=self.names)
        self.feature_importance_unique.append(_feature_importance.values.T)
        feature_importance["step"] = range(
            -past_feature_importance.shape[0],
            future_feature_importance.shape[0],
        )
        patch_importance = self.calculate_patch_importance(feature_importance, features)
        return y_ans, y, feature_importance, patch_importance

    def calc_important_parts(self, feature_importance, patch_importance, k):
        if "point" in self.method:  # TFT-point
            important_parts, _ = self.extract_important_features(feature_importance, k)
        elif "patch" in self.method:  # TFT-patch
            important_parts = self.extract_important_patches(patch_importance, k)
        return important_parts

    def extract_important_features(self, feature_importance, k):
        # Sort features based on feature importance
        feature_importance = feature_importance.drop("step", axis=1)
        feature_importance_long = feature_importance.stack().reset_index()
        feature_importance_long.columns = ["index", "column", "value"]
        sorted_feature_importance = feature_importance_long.sort_values(
            by="value", ascending=False
        )
        # Extract the top k% features
        important_features_num = int(len(sorted_feature_importance) * k // 100)
        important_features = sorted_feature_importance.head(important_features_num)[
            ["index", "column"]
        ]
        other_features = sorted_feature_importance.tail(
            len(sorted_feature_importance) - important_features_num
        )[["index", "column"]]
        return important_features, other_features

    def calculate_patch_importance(self, feature_importance, features):
        feature_importance = feature_importance.rename(columns={"observed_target": "y"})
        patch_importance = {}
        for i, (begin, end) in enumerate(zip(self.patch_begins, self.patch_ends)):
            subset = feature_importance[
                (feature_importance["step"] >= begin)
                & (feature_importance["step"] < end)
            ]
            for col in feature_importance.columns:
                tag = f"{col}_{i}"
                if tag in features:
                    patch_importance[tag] = subset[col].sum().item()
        return patch_importance

    def replace_values(self, df, idx, removed_var):
        # Values after replacement
        ## "Local" focuses only on values in the patch
        ## "Global" uses all values in test dataset
        if self.replace == "zero":
            replacing_vals = 0
        elif self.replace == "local-mean":
            local_mean = self.df_window[removed_var].mean()
            replacing_vals = local_mean
        elif self.replace == "global-mean":
            global_mean = self.df_unique[removed_var].tail(self.test_size).mean()
            replacing_vals = global_mean
        # Replace the patch
        if "point" in self.method:
            df.loc[idx, removed_var] = replacing_vals
        elif "patch" in self.method:
            x0 = self.patch_begins[idx]
            x1 = self.patch_ends[idx]
            df.loc[
                (df["step"] >= x0) & (df["step"] <= x1),
                removed_var,
            ] = replacing_vals
        return df

    # Override the function "prepare" in VisualizerPatchDecomp
    def prepare(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
    ):
        unique_ids = list(df["unique_id"].unique())

        # datetimes and forecasting cutoffs
        datetimes, cutoffs = {}, {}
        for uid in unique_ids:
            datetimes[uid] = df.loc[df["unique_id"] == uid, "ds"].unique()
            cutoffs[uid] = datetimes[uid][self.input_size - 1 : -self.horizon]

        self.df = df
        self.static_df = static_df
        self.datetimes = datetimes
        self.cutoffs = cutoffs
        self.unique_ids = unique_ids


class LocalExplanationDrawer(AOPCRCalculatorPatchDecomp):
    def __init__(self, method: str, dirname: str, seed: int):
        super().__init__(method, dirname, seed)

    def simulate(self, test_size, path):
        self.test_size = test_size
        self.path = path
        features = self.feature_names()
        color_scaler = 10
        ys_ans, ys = [], []

        with suppress_output_except_tqdm() as tqdm_out_stream:
            # Calculate by unique ID
            for unique_id in self.unique_ids:
                self.y_decomps_unique = []
                self.df_unique = self.df[self.df["unique_id"] == unique_id].reset_index(
                    drop=True
                )
                self.data_length = self.df_unique.shape[0]
                with tqdm(
                    range(
                        self.data_length - self.test_size,
                        self.data_length - self.horizon,
                    ),
                    file=tqdm_out_stream,
                ) as pbar_steps:
                    for index in pbar_steps:
                        # Control timing to plot
                        if index % 1000 != 0:
                            # if index - self.input_size + 1 != 49933:
                            continue

                        pbar_steps.set_description(f"{unique_id} ")
                        begin = index - self.input_size + 1
                        end = index + self.horizon + 1
                        self.df_window = self.df_unique.iloc[begin:end].reset_index(
                            drop=True
                        )

                        # Predict
                        y_ans, y, _, patch_importance = self.predict_wo_removal(
                            features
                        )
                        ys_ans.append(y_ans)

                        # Adjust color intensity
                        scaler = (
                            np.nansum(np.array(list(patch_importance.values())))
                            * color_scaler
                        )
                        patch_importance = {
                            key: value / scaler * len(patch_importance)
                            for key, value in patch_importance.items()
                            if not np.isnan(value)
                        }

                        # Plot
                        self.plot_patch_importance_w_timeseries(
                            y[0],
                            patch_importance,
                            self.path + f"/local_explanation_{begin}.pdf",
                        )
                        self.plot_decomposed_timeseries(
                            y[0],
                            self.path + f"/decomposition_{begin}.svg",
                        )
                        self.plot_decomposed_timeseries_top_k(
                            y[0],
                            patch_importance,
                            8,
                            self.path + f"/decomposition_{begin}_selected.svg",
                        )

    def plot_patch_importance_w_timeseries(self, y, patch_importance, path):
        fig = plt.figure(figsize=(5, 4))
        for row, feature_name in enumerate(self.names):
            ax = fig.add_subplot(len(self.names), 1, row + 1)
            xticks = []
            for patch in range(self.n_patches):
                name = f"{feature_name}_{patch}"
                if name in self.patch_names:
                    x0 = self.patch_begins[patch]
                    xticks.append(x0)
                    x1 = self.patch_ends[patch]
                    ax.axvspan(
                        x0, x1, alpha=min(patch_importance[name], 1), color="tab:red"
                    )
            plt.plot(self.t, self.df_window[feature_name], c="black")
            if feature_name == "y":
                plt.plot(self.t_futr, y, c="tab:red")
            xticks.append(x1)
            ax.set_xlim(self.t[0] - 5, self.t[-1] + 5)
            ax.text(
                -0.12,
                0.5,
                feature_name,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xticks(xticks)
        # plt.suptitle("Local patch importance")
        plt.tight_layout(h_pad=0.5)
        plt.savefig(path)
        plt.close()

    def plot_decomposed_timeseries(self, y, path):
        y_decomp = self.model.y_decomp
        ax2s = []
        ax2_ylim = 0
        fig = plt.figure(figsize=(5.5, 5))
        for row, feature_name in enumerate(self.names):
            ax1 = fig.add_subplot(len(self.names), 5, (row * 5 + 1, row * 5 + 4))
            ax2 = fig.add_subplot(len(self.names), 5, (row + 1) * 5)
            y_feature = []
            xticks, labels, colors = [], [], []
            for patch in range(self.n_patches):
                name = f"{feature_name}_{patch}"
                if name in self.patch_names:
                    y_feature.append(y_decomp[0, self.patch_names.index(name), 0])
                    color = px.colors.qualitative.D3[(patch) % 10]
                    x0 = self.patch_begins[patch]
                    xticks.append(x0)
                    x1 = self.patch_ends[patch]
                    ax1.axvspan(x0, x1, alpha=0.3, color=color)
                    labels.append(f"patch-{patch} ({x0 + 1} ~ {x1})")
                    colors.append(color)
            ax1.plot(self.t, self.df_window[feature_name], c="black")
            if feature_name == "y":
                ax1.plot(self.t_futr, y, c="tab:red")
                ax2.set_title("decomp.")
            xticks.append(x1)
            ax1.set_xlim(self.t[0] - 5, self.t[-1] + 5)
            ax1.set_title(feature_name)
            ax1.set_xticks([])
            ax2.stackplot(
                self.t_futr, y_feature, labels=labels, colors=colors, alpha=0.3
            )
            ax2.plot(self.t_futr, sum(y_feature), c="tab:red")
            ax2.set_xticks([])
            ax2s.append(ax2)
            ax2_ylim = max(ax2_ylim, abs(plt.ylim()[0]), abs(plt.ylim()[1]))
            ax1.grid(axis="y")
            ax2.grid(axis="y")
        ax1.set_xticks(xticks)
        ax2.set_xticks([x for x in xticks if x >= 0])
        for ax2 in ax2s:
            ax2.set_ylim(-ax2_ylim * 1.2, ax2_ylim * 1.2)
        # plt.suptitle("Contribution Decomposition")
        plt.tight_layout(h_pad=0.5)
        plt.savefig(path)
        plt.close()

    def plot_decomposed_timeseries_top_k(self, y, patch_importance, k, path):
        y_decomp = self.model.y_decomp
        important_patches = self.extract_important_patches(patch_importance, k)
        ax2s = []
        ax2_ylim = 0
        fig = plt.figure(figsize=(5.5, 5))
        for row, feature_name in enumerate(self.names):
            ax1 = fig.add_subplot(len(self.names), 8, (row * 8 + 1, row * 8 + 7))
            ax2 = fig.add_subplot(len(self.names), 8, (row + 1) * 8)
            y_feature, y_feature_selected = [], []
            xticks, labels, colors = [], [], []
            for patch in range(self.n_patches):
                name = f"{feature_name}_{patch}"
                if name in self.patch_names:
                    y_feature.append(y_decomp[0, self.patch_names.index(name), 0])
                    if name in important_patches:
                        color = "tab:red"
                    else:
                        color = "tab:blue"
                    x0 = self.patch_begins[patch]
                    xticks.append(x0)
                    x1 = self.patch_ends[patch]
                    ax1.axvspan(x0, x1, alpha=0.3, color=color)
                    y_feature_selected.append(
                        y_decomp[0, self.patch_names.index(name), 0]
                    )
                    labels.append(f"patch-{patch} ({x0 + 1} ~ {x1})")
                    colors.append(color)
            ax1.plot(self.t, self.df_window[feature_name], c="black")
            if feature_name == "y":
                ax1.plot(self.t_futr, y, c="tab:red")
                ax2.set_title("decomp.")
            xticks.append(x1)
            ax1.set_xlim(self.t[0] - 5, self.t[-1] + 5)
            ax1.set_title(feature_name)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if len(y_feature_selected) > 0:
                ax2.stackplot(
                    self.t_futr,
                    y_feature_selected,
                    labels=labels,
                    colors=colors,
                    alpha=0.3,
                )
            ax2.plot(self.t_futr, sum(y_feature), c="tab:red")
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2s.append(ax2)
            ax2_ylim = max(ax2_ylim, abs(plt.ylim()[0]), abs(plt.ylim()[1]))
        ax1.set_xticks(xticks)
        ax2.set_xticks([x for x in xticks if x >= 0])
        for ax2 in ax2s:
            ax2.set_ylim(-ax2_ylim * 1.2, ax2_ylim * 1.2)
        # plt.suptitle("Contribution Decomposition")
        plt.tight_layout(h_pad=0.5)
        plt.savefig(path)
        plt.close()
