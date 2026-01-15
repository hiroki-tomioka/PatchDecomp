import sys
import os
import contextlib
import json
import pandas as pd
import datetime as dt
from ray import tune
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info
from epftoolbox.data import read_data


# Colors for visualization
MODEL_COLORS = {
    "PatchDecomp": "tab:red",
    "PatchTST": "tab:blue",
    "NBEATS": "tab:orange",
    "NBEATSx": "tab:orange",
    "NHITS": "tab:green",
    "TFT": "tab:purple",
    "DLinear": "tab:brown",
    "TSMixer": "tab:pink",
    "TSMixerx": "tab:pink",
    "Autoformer": "tab:gray",
    "iTransformer": "tab:olive",
    "TiDE": "tab:cyan",
}


def ray2optuna(ray_config):
    """
    This function is adapted from the Apache 2.0 licensed project Nixtla/neuralforecast:
    https://github.com/Nixtla/neuralforecast/blob/v1.6.4/neuralforecast/common/_base_auto.py#L262
    Original copyright: (c) 2024 Nixtla
    Modifications made by [Your Name] in 2025

    Convert ray config dict into optuna config function.
    """

    def optuna_config(trial):
        out = {}
        for k, v in ray_config.items():
            if hasattr(v, "sampler"):
                sampler = v.sampler
                if isinstance(sampler, tune.search.sample.Integer.default_sampler_cls):
                    v = trial.suggest_int(k, v.lower, v.upper)
                elif isinstance(
                    sampler, tune.search.sample.Categorical.default_sampler_cls
                ):
                    v = trial.suggest_categorical(k, v.categories)
                elif isinstance(sampler, tune.search.sample.Uniform):
                    # v = trial.suggest_uniform(k, v.lower, v.upper)
                    v = trial.suggest_float(k, v.lower, v.upper)
                elif isinstance(sampler, tune.search.sample.LogUniform):
                    # v = trial.suggest_loguniform(k, v.lower, v.upper)
                    v = trial.suggest_float(k, v.lower, v.upper, log=True)
                elif isinstance(sampler, tune.search.sample.Quantized):
                    if isinstance(
                        sampler.get_sampler(), tune.search.sample.Float._LogUniform
                    ):
                        v = trial.suggest_float(k, v.lower, v.upper, log=True)
                    elif isinstance(
                        sampler.get_sampler(), tune.search.sample.Float._Uniform
                    ):
                        v = trial.suggest_float(k, v.lower, v.upper, step=sampler.q)
                else:
                    raise ValueError(f"Coudln't translate {type(v)} to optuna.")
            out[k] = v
        return out

    return optuna_config


@contextlib.contextmanager
def suppress_output_except_tqdm():
    """
    Suppress standard output other than the tqdm display
    """
    # File descriptor for saving
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_fd_out = os.dup(1)
    saved_fd_err = os.dup(2)

    try:
        # Create a new stream for the output of tqdm
        tqdm_stream = open(os.devnull, "w")
        tqdm_out = os.dup(1)
        os.dup2(tqdm_stream.fileno(), 1)
        os.dup2(tqdm_stream.fileno(), 2)
        sys.stdout = tqdm_stream
        sys.stderr = tqdm_stream

        # Set the output of tqdm to the original standard output
        tqdm_out_stream = os.fdopen(tqdm_out, "w")
        yield tqdm_out_stream

    finally:
        # Restore the original file descriptor
        os.dup2(saved_fd_out, 1)
        os.dup2(saved_fd_err, 2)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.close(saved_fd_out)
        os.close(saved_fd_err)
        tqdm_stream.close()


def load_csv(path):
    """
    Load a csv file

    Args:
        path (str): path to the csv file

    Returns:
        (pd.DataFrame): loaded dataframe
    """
    df = pd.read_csv(path, index_col=0)
    return df


def make_folder(base_dir, suffix=None):
    """
    Make a folder

    Args:
        base_dir (str): path just above the folder to be created
        suffix (str, optional): folder suffix. Defaults to None.

    Returns:
        (str): path to created folder
    """
    now = dt.datetime.now()
    save_dir = base_dir + now.strftime("%y%m%d_%H%M%S")
    if suffix is None:
        save_dir += "/"
    else:
        save_dir += f"_{suffix}/"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_args(params, path):
    """
    Save experimental parameters

    Args:
        params (dict): experimental parameters
        path (str): path to save
    """
    with open(path + "args.json", mode="w") as f:
        json.dump(params, f, indent=2)


def load_ltsf_dataset(dataset):
    """
    Load LTSF dataset

    Args:
        dataset (str): dataset name (ex. ETTh1, Weather)

    Returns:
        df (pd.DataFrame): data
    """
    df = LongHorizon2.load(directory="../../datasets/", group=dataset)
    df["ds"] = pd.to_datetime(df["ds"])
    return df, None


def load_epf_dataset(unique_ids):
    """
    Load EPF dataset
    https://epftoolbox.readthedocs.io/en/latest/modules/data_extract.html

    Args:
        unique_ids (list): unique IDs to use

    Returns:
        df (pd.DataFrame): data
        static_df (pd.DataFrame or None): static variables
    """
    dfs = []
    for unique_id in unique_ids:
        df_train, df_test = read_data(
            path="../../datasets/epftoolbox/", dataset=unique_id
        )
        df_all = pd.concat([df_train, df_test]).reset_index()
        df_all = df_all.rename(
            columns={
                "Date": "ds",
                "index": "ds",  # Only DE does not have index "Date"
                "Price": "y",
            }
        )
        df_all["ds"] = pd.to_datetime(df_all["ds"])
        df_all["month"] = df_all["ds"].dt.month
        df_all["week_day"] = df_all["ds"].dt.dayofweek
        df_all["hour"] = df_all["ds"].dt.hour
        df_all["unique_id"] = unique_id
        dfs.append(df_all)
    df = pd.concat(dfs).reset_index(drop=True)
    if unique_ids == ["PJM", "NP", "BE", "FR", "DE"]:
        static_df = pd.DataFrame(
            [["PJM", 0], ["NP", 1], ["BE", 2], ["FR", 3], ["DE", 4]],
            columns=["unique_id", "market_id"],
        )
    elif len(unique_ids) == 1:
        static_df = None
    else:
        raise NotImplementedError
    return df, static_df
