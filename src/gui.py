import sys
import glob
import numpy as np
import pandas as pd
from IPython.display import display
from neuralforecast import NeuralForecast, core
from neuralforecast.tsdataset import TimeSeriesDataset
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import (
    Dropdown,
    SelectionSlider,
    BoundedIntText,
    Play,
    HBox,
    VBox,
    Output,
    jslink,
)

sys.path.append("../src")
from models.patchdecomp import PatchDecomp


class Visualizer:
    pass


class VisualizerPatchDecomp(Visualizer):
    """
    Visualize the input/predicted/decomposed data of PatchDecomp
    """

    def __init__(
        self,
        dirname: str,
    ):

        # Load model
        core.MODEL_FILENAME_DICT["autopatchdecomp"] = PatchDecomp
        nf = NeuralForecast.load(dirname, weights_only=False)
        model = nf.models[0]
        model.set_test_size(nf.h)

        self.nf = nf
        self.model = model
        self.horizon = model.h
        self.input_size = model.input_size

    def prepare(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
    ):
        unique_ids = list(df["unique_id"].unique())

        # Datetimes and forecasting cutoffs
        datetimes, cutoffs = {}, {}
        for uid in unique_ids:
            datetimes[uid] = df.loc[df["unique_id"] == uid, "ds"].unique()
            cutoffs[uid] = datetimes[uid][self.input_size - 1 : -self.horizon]

        self.df = df
        self.static_df = static_df
        self.datetimes = datetimes
        self.cutoffs = cutoffs
        self.unique_ids = unique_ids

        # Initial prediction
        self.df_window, self.y, self.y_decomp = self.predict(
            unique_ids[0], cutoffs[unique_ids[0]][0]
        )

    def predict(
        self,
        unique_id,
        cutoff,
    ):

        index = list(self.datetimes[unique_id]).index(cutoff)
        begin = self.datetimes[unique_id][index - self.input_size + 1]
        end = self.datetimes[unique_id][index + self.horizon]

        # Make dataset
        df = self.df
        df_window = df.loc[
            (df["ds"] >= begin) & (df["ds"] <= end) & (df["unique_id"] == unique_id)
        ]
        dataset, *_ = TimeSeriesDataset.from_df(df_window, static_df=self.static_df)

        # Predict
        y = self.model.decompose(dataset=dataset)
        y_decomp = self.model.y_decomp

        return df_window, y, y_decomp

    def create(
        self,
        width: int = 1500,
        height: int = 1000,
    ):

        model = self.model
        names = ["y"] + model.hist_exog_list + model.futr_exog_list

        # Create widgets
        dropdown = Dropdown(
            description="unique_id",
            options=self.unique_ids,
            layout=dict(width="300px"),
        )
        slider = SelectionSlider(
            description="cutoff",
            options=self.cutoffs[dropdown.value],
            layout=dict(width="1000px"),
        )
        play = Play(
            value=0,
            min=0,
            max=len(self.cutoffs[dropdown.value]) - 1,
            step=1,
            interval=1000,
        )
        step = BoundedIntText(
            description="step",
            value=1,
            min=1,
            layout=dict(width="150px"),
        )
        interval = BoundedIntText(
            description="interval",
            value=1000,
            min=100,
            max=10000,
            layout=dict(width="150px"),
        )
        out = Output(
            layout=dict(height="80px"),
        )
        trash = Output(
            layout=dict(display="none"),
        )

        # Make figure
        fig = make_subplots(
            rows=len(names),
            cols=2,
            shared_xaxes=True,
            column_widths=[self.input_size + self.horizon, self.horizon],
            horizontal_spacing=0.03,
            vertical_spacing=0.05,
            subplot_titles=sum([[name, ""] for name in names], []),
            column_titles=["", "decomposition"],
        )

        # Plot actual time series
        for row, name in enumerate(names):
            if name in ["y"] + model.futr_exog_list:
                y_ = self.df_window[name]
                t_ = model.t
            else:
                y_ = self.df_window.iloc[: self.input_size][name]
                t_ = model.t_hist
            fig.add_trace(
                go.Scatter(
                    x=t_,
                    y=y_,
                    line=dict(color="black"),
                    showlegend=False,
                ),
                row=row + 1,
                col=1,
            )
            fig.update_xaxes(
                row=row + 1,
                col=1,
                dtick=model.patch_size,
                tick0=0,
            )

        # Plot predicted time series
        fig.add_trace(
            go.Scatter(
                x=model.t_futr,
                y=self.y[0],
                line=dict(color="red"),
                name="prediction",
            ),
            row=1,
            col=1,
        )

        # Plot area chart
        ymax = np.abs(self.y_decomp[0].sum(axis=0)).max() * 1.1
        y_feature_all = 0
        for row, feature_name in enumerate(names):
            fig.update_xaxes(
                row=row + 1,
                col=2,
                dtick=model.patch_size,
            )
            fig.update_yaxes(
                range=[-ymax, ymax],
                row=row + 1,
                col=2,
            )
            if feature_name in model.time_names:
                continue
            y_feature = 0
            showlegend = (
                feature_name
                == [name for name in names if name not in model.time_names][-1]
            )
            for patch in range(model.n_patches):
                name = f"{feature_name}_{patch}"
                if name in model.patch_names:
                    y_feature += self.y_decomp[0, model.patch_names.index(name), 0]
                    y_feature_all += self.y_decomp[0, model.patch_names.index(name), 0]
                    color = px.colors.qualitative.D3[(patch) % 10]
                    x0 = model.patch_begins[patch]
                    x1 = model.patch_ends[patch]
                    fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        line_width=0,
                        fillcolor=color,
                        opacity=0.4,
                        layer="below",
                        showlegend=False,
                        row=row + 1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=model.t_futr,
                            y=y_feature,
                            mode="lines",
                            fill="tonexty",
                            name=f"patch-{patch} ({x0 + 1} ~ {x1})",
                            line=dict(color=color, width=0),
                            legendgroup=patch,
                            showlegend=showlegend,
                        ),
                        row=row + 1,
                        col=2,
                    )
            fig.add_trace(
                go.Scatter(
                    x=model.t_futr,
                    y=y_feature,
                    mode="lines",
                    line=dict(color="red"),
                    showlegend=False,
                ),
                row=row + 1,
                col=2,
            )

        fig.update_layout(
            width=width,
            height=height,
        )

        # Make figure widget
        g = go.FigureWidget(fig)

        # Event listener
        def update_fig(change):
            unique_id = dropdown.value
            cutoff = slider.value

            # Update unique_id
            update_uid = False
            if change:
                if change.owner == dropdown:
                    update_uid = True
            else:
                update_uid = True
            if update_uid and self.static_df is not None:
                out.clear_output()
                with out:
                    display(
                        self.static_df.loc[self.static_df["unique_id"] == unique_id]
                    )

            # Predict
            with trash:
                df_window, y, y_decomp = self.predict(unique_id, cutoff)
            trash.clear_output()
            df_window = df_window.set_index("ds")
            g.layout.title = f"{df_window.index[0]} ~ {df_window.index[-1]}"
            ymax = np.abs(y_decomp[0].sum(axis=0)).max() * 1.1

            # Update data
            with g.batch_update():
                trace = 0
                for row, name in enumerate(names):
                    if name in ["y"] + model.futr_exog_list:
                        y_ = df_window[name]
                    else:
                        y_ = df_window.iloc[: self.input_size][name]
                    g.data[trace].y = y_
                    trace += 1
                g.data[trace].y = y[0]
                trace += 1
                for row, feature_name in enumerate(names):
                    g.update_yaxes(
                        range=[-ymax, ymax],
                        row=row + 1,
                        col=2,
                    )
                    if feature_name in model.time_names:
                        continue
                    y_feature = 0
                    for patch in range(model.n_patches):
                        name = f"{feature_name}_{patch}"
                        if name in model.patch_names:
                            y_feature += y_decomp[0, model.patch_names.index(name), 0]
                            g.data[trace].y = y_feature
                            trace += 1
                    g.data[trace].y = y_feature
                    trace += 1

        def update_step(change):
            play.step = step.value

        def update_interval(change):
            play.interval = interval.value

        # Event handling
        dropdown.observe(update_fig, names="value")
        slider.observe(update_fig, names="value")
        step.observe(update_step, names="value")
        interval.observe(update_interval, names="value")
        jslink((play, "value"), (slider, "index"))
        update_fig({})

        # Return GUI
        return VBox(
            [
                HBox([dropdown, out]),
                HBox([play, slider, step, interval]),
                g,
                trash,
            ]
        )


class VisualizerAll(Visualizer):
    """
    Visualize the input/predicted data of all methods
    """

    def __init__(
        self,
        dirname: str,
    ):

        core.MODEL_FILENAME_DICT["autopatchdecomp"] = PatchDecomp

        model_dirs = sorted(glob.glob(dirname + "/*"))

        # Load models
        nfs = {}
        for model_dir in model_dirs:
            try:
                loaded_model = NeuralForecast.load(model_dir, weights_only=False)
                nfs[model_dir.split("/")[-1]] = loaded_model
            except RuntimeError:
                continue
        models = {}
        for model_name, nf in nfs.items():
            model = nf.models[0]
            model.set_test_size(nf.h)
            models[model_name] = model
            horizon = model.h
            input_size = model.input_size

        self.nfs = nfs
        self.models = models
        self.horizon = horizon
        self.input_size = input_size

    def prepare(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
    ):
        unique_ids = list(df["unique_id"].unique())

        # Datetimes and forecasting cutoffs
        datetimes, cutoffs = {}, {}
        for uid in unique_ids:
            datetimes[uid] = df.loc[df["unique_id"] == uid, "ds"].unique()
            cutoffs[uid] = datetimes[uid][self.input_size - 1 : -self.horizon]

        self.df = df
        self.static_df = static_df
        self.datetimes = datetimes
        self.cutoffs = cutoffs
        self.unique_ids = unique_ids

        # Initial prediction
        self.df_window, self.ys = self.predict(unique_ids[0], cutoffs[unique_ids[0]][0])

    def predict(
        self,
        unique_id,
        cutoff,
    ):
        index = list(self.datetimes[unique_id]).index(cutoff)
        begin = self.datetimes[unique_id][index - self.input_size + 1]
        end = self.datetimes[unique_id][index + self.horizon]

        # Make dataset
        df = self.df
        df_window = df.loc[
            (df["ds"] >= begin) & (df["ds"] <= end) & (df["unique_id"] == unique_id)
        ]
        dataset, *_ = TimeSeriesDataset.from_df(df_window, static_df=self.static_df)

        # Predict
        ys = {}
        for model_name, model in self.models.items():
            y = model.predict(dataset=dataset)
            ys[model_name] = y

        return df_window, ys

    def create(
        self,
        width: int = 1500,
        height: int = 1000,
    ):
        names = list(self.models.keys())

        # Create widgets
        dropdown = Dropdown(
            description="unique_id",
            options=self.unique_ids,
            layout=dict(width="300px"),
        )
        slider = SelectionSlider(
            description="cutoff",
            options=self.cutoffs[dropdown.value],
            layout=dict(width="1000px"),
        )
        play = Play(
            value=0,
            min=0,
            max=len(self.cutoffs[dropdown.value]) - 1,
            step=1,
            interval=1000,
        )
        step = BoundedIntText(
            description="step",
            value=1,
            min=1,
            layout=dict(width="150px"),
        )
        interval = BoundedIntText(
            description="interval",
            value=1000,
            min=100,
            max=10000,
            layout=dict(width="150px"),
        )
        out = Output(
            layout=dict(height="80px"),
        )
        trash = Output(
            layout=dict(display="none"),
        )

        # Make figure
        fig = make_subplots(
            rows=len(names),
            cols=1,
            shared_xaxes=True,
            column_widths=[self.input_size + self.horizon],
            horizontal_spacing=0.03,
            vertical_spacing=0.05,
            subplot_titles=[name for name in names],
            column_titles=[""],
        )

        t_hist = np.arange(self.input_size) - self.input_size + 1
        t_futr = np.arange(self.horizon) + 1
        t_ = np.concatenate([t_hist, t_futr])

        # Plot time series
        for row, name in enumerate(names):
            y_ = self.df_window["y"]
            fig.add_trace(
                go.Scatter(
                    x=t_,
                    y=y_,
                    line=dict(color="black"),
                    showlegend=False,
                ),
                row=row + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=t_futr,
                    y=self.ys[name][:, 0],
                    line=dict(color="red"),
                    name="prediction",
                ),
                row=row + 1,
                col=1,
            )
            fig.update_xaxes(
                row=row + 1,
                col=1,
                tick0=0,
            )

        fig.update_layout(
            width=width,
            height=height,
        )

        # Make figure widget
        g = go.FigureWidget(fig)

        # Event listener
        def update_fig(change):
            unique_id = dropdown.value
            cutoff = slider.value

            # Update unique_id
            update_uid = False
            if change:
                if change.owner == dropdown:
                    update_uid = True
            else:
                update_uid = True
            if update_uid and self.static_df is not None:
                out.clear_output()
                with out:
                    display(
                        self.static_df.loc[self.static_df["unique_id"] == unique_id]
                    )

            # Predict
            with trash:
                df_window, ys = self.predict(unique_id, cutoff)
            trash.clear_output()
            df_window = df_window.set_index("ds")
            g.layout.title = f"{df_window.index[0]} ~ {df_window.index[-1]}"

            # Update data
            with g.batch_update():
                trace = 0
                for row, name in enumerate(names):
                    y_ = df_window["y"]
                    g.data[trace].y = y_
                    trace += 1

                    g.data[trace].y = ys[name][:, 0]
                    trace += 1

        def update_step(change):
            play.step = step.value

        def update_interval(change):
            play.interval = interval.value

        # Event handling
        dropdown.observe(update_fig, names="value")
        slider.observe(update_fig, names="value")
        step.observe(update_step, names="value")
        interval.observe(update_interval, names="value")
        jslink((play, "value"), (slider, "index"))
        update_fig({})

        # Return GUI
        return VBox(
            [
                HBox([dropdown, out]),
                HBox([play, slider, step, interval]),
                g,
                trash,
            ]
        )
