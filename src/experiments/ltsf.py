import os
import sys
import pathlib
import time
import yaml
import argparse
import logging
import optuna
import pandas as pd
import datetime as dt
import torch
import pytorch_lightning as pl
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoNBEATS,
    AutoNBEATSx,
    AutoNHITS,
    AutoAutoformer,
    AutoPatchTST,
    AutoTFT,
    AutoDLinear,
    AutoTSMixerx,
    AutoiTransformer,
    AutoTiDE,
)
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mae, mse
from ray import tune
from datasetsforecast.long_horizon2 import LongHorizon2Info

sys.path.append("..")
from models.patchdecomp import AutoPatchDecomp
from utils import ray2optuna, load_ltsf_dataset


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.backends.cudnn.deterministic = True


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# Setting
now = dt.datetime.now().strftime("%y%m%d_%H%M%S")
rootpath = pathlib.Path(f"../../results/ltsf/seed{args.seed}_{now}/")
dataset_names = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'TrafficL']
horizons = [96, 192, 336, 720]
input_sizes = [512]
max_epochs = 20
early_stop_patience_steps = 5
n_trials = 10
scaler_type = "standard"
batch_size = 8
windows_batch_size = 128
learning_rate = 1e-3
random_seed = args.seed
backend = "optuna"

# Based on TSMixer's experimental setup
# https://arxiv.org/abs/2303.06053
experiments = {}
for dataset_name in dataset_names:
    val_size = 0.2
    test_size = 0.2 if "ETT" in dataset_name else 0.1
    for horizon in horizons:
        for input_size in input_sizes:
            experiment = dict(
                dataset_name=dataset_name,
                horizon=horizon,
                input_size=input_size,
                val_size=val_size,
                test_size=test_size,
                # step_size=horizon,  # error: `test_size - h` should be module `step_size`
                step_size=1,
                loss=MAE(),
            )
            experiments[f"{dataset_name}_{horizon}"] = experiment

# Model configs
config_models = dict(
    PatchDecomp=dict(
        class_name="AutoPatchDecomp",
        patch_size=tune.choice([12, 24, 48]),
        hidden_size=tune.choice([32, 64, 128, 256]),
        n_head=tune.choice([4, 8]),
        encoder_layers=tune.choice([1, 2, 3, 4]),
        dropout=tune.uniform(0.0, 0.5),
    ),
    PatchTST=dict(
        class_name="AutoPatchTST",
        hidden_size=tune.choice([32, 64, 128, 256]),
        n_heads=tune.choice([4, 8]),
        patch_len=tune.choice([12, 24, 48]),
        stride=12,
        dropout=tune.uniform(0.0, 0.5),
        encoder_layers=3,
    ),
    NBEATS=dict(
        class_name="AutoNBEATSx",
        n_harmonics=2,
        n_polynomials=2,
        stack_types=["identity", "trend", "seasonality"],
        n_blocks=[1, 1, 1],
        mlp_units=tune.choice(
            [
                3 * [[32, 32]],
                3 * [[64, 64]],
                3 * [[128, 128]],
                3 * [[256, 256]],
            ]
        ),
        dropout_prob_theta=tune.uniform(0.0, 0.5),
    ),
    NHITS=dict(
        class_name="AutoNHITS",
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        mlp_units=tune.choice(
            [
                3 * [[32, 32]],
                3 * [[64, 64]],
                3 * [[128, 128]],
                3 * [[256, 256]],
            ]
        ),
        dropout_prob_theta=tune.uniform(0.0, 0.5),
    ),
    TFT=dict(
        class_name="AutoTFT",
        hidden_size=tune.choice([32, 64, 128, 256]),
        n_head=tune.choice([4, 8]),
        dropout=tune.uniform(0.0, 0.5),
    ),
    DLinear=dict(
        class_name="AutoDLinear",
        moving_avg_window=tune.choice([11, 25, 51]),
    ),
    TSMixer=dict(
        class_name="AutoTSMixerx",
        n_series=1,
        n_block=tune.choice([1, 2, 4, 6, 8]),
        ff_dim=tune.choice([32, 64, 128, 256]),
        dropout=tune.uniform(0.0, 0.5),
    ),
    Autoformer=dict(
        class_name="AutoAutoformer",
        hidden_size=tune.choice([32, 64, 128, 256]),
        dropout=tune.uniform(0.0, 0.5),
        n_head=tune.choice([4, 8]),
    ),
    iTransformer=dict(
        class_name="AutoiTransformer",
        n_series=1,
        hidden_size=tune.choice([32, 64, 128, 256]),
        dropout=tune.uniform(0.0, 0.5),
        n_heads=tune.choice([4, 8]),
        e_layers=tune.choice([1, 2, 3, 4]),
        d_layers=tune.choice([1, 2, 3, 4]),
    ),
    TiDE=dict(
        class_name="AutoTiDE",
        hidden_size=tune.choice([32, 64, 128, 256]),
        num_encoder_layers=tune.choice([1, 2, 3, 4]),
        num_decoder_layers=tune.choice([1, 2, 3, 4]),
        dropout=tune.uniform(0.0, 0.5),
    ),
)

# Experiment loop
for experiment_name, experiment in experiments.items():
    dataset_name = experiment["dataset_name"]
    horizon = experiment["horizon"]
    input_size = experiment["input_size"]

    # Load dataset
    df, _ = load_ltsf_dataset(dataset_name)

    # Dataset information
    dataset_info = LongHorizon2Info[dataset_name]
    freq = dataset_info.freq
    if dataset_name == "Weather":
        freq = "10min"  # 10M -> 10min (10M = 10 months)
    val_size = experiment.get("val_size", dataset_info.val_size)
    test_size = experiment.get("test_size", dataset_info.test_size)
    step_size = experiment["step_size"]
    length = len(df.ds.unique())
    if type(val_size) is float:
        val_size = int(length * val_size)
    if type(test_size) is float:
        test_size = int(length * test_size)

    # Model loop
    for model_name, config_model in config_models.items():

        # Adjust maximum epoch to be max_epochs when early stopping is not applied
        num_features = len(df.unique_id.unique())
        if "n_series" in config_model:
            iteration = (num_features - 1) // config_model["n_series"] + 1
        else:
            iteration = (num_features - 1) // batch_size + 1
        max_steps = iteration * max_epochs
        val_check_steps = iteration

        # Common config
        config = dict(
            input_size=input_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            random_seed=random_seed,
            scaler_type=scaler_type,
            batch_size=batch_size,
            logger=[
                pl.loggers.TensorBoardLogger(
                    save_dir=rootpath,
                    name="logs",
                    version=f"{experiment_name}/{model_name}",
                ),
            ],
        )

        # Update config
        config.update(config_model)
        class_name = config.pop("class_name")
        if "n_series" in config:
            n_series = config["n_series"]
        else:
            config["windows_batch_size"] = windows_batch_size
            config["valid_batch_size"] = batch_size
            config["inference_windows_batch_size"] = windows_batch_size
        print("#" * 64)
        print(class_name)
        if backend == "optuna":
            config = ray2optuna(config)

        # Create model instance
        model_kws = dict(
            h=horizon,
            loss=experiment["loss"],
            config=config,
            search_alg=optuna.samplers.TPESampler(seed=random_seed),
            num_samples=n_trials,
            backend=backend,
        )
        if ("AutoTSMixer" in class_name) or (
            "AutoiTransformer" in class_name
        ):  # n_series is a required argument
            model_kws["n_series"] = n_series
        model = eval(class_name)(**model_kws)
        nf = NeuralForecast(
            models=[model],
            freq=freq,
        )

        # Train, valid, test
        t0 = time.time()
        try:
            cv_df = nf.cross_validation(
                df=df,
                val_size=val_size,
                test_size=test_size,
                step_size=step_size,
                n_windows=None,
            )
        except Exception as e:
            print("#" * 64)
            print("Fail")
            print(experiment_name, class_name)
            print(e)
            continue
        elapsed_time = time.time() - t0
        print("#" * 64)
        print("Success")
        print(experiment_name, class_name)
        print(cv_df)

        # Make directory
        save_dirpath = rootpath / "results" / experiment_name / model_name
        save_dirpath.mkdir(parents=True, exist_ok=True)

        # Save model
        nf.save(
            path=str(save_dirpath),  # Path -> str
            model_index=None,
            overwrite=True,
            save_dataset=False,
        )

        # Save parameter tuning history and the best parameter
        results = nf.models[0].results
        if backend == "ray":
            df_tune = results.get_dataframe()
            best_config = results.get_best_result().config
        elif backend == "optuna":
            df_tune = results.trials_dataframe()
            best_config = results.best_params
        df_tune.to_csv(save_dirpath / "tune.csv", index=False)
        with open(save_dirpath / "best_config.yml", "w") as f:
            yaml.dump(best_config, f)

        # Save metric
        y_true = cv_df["y"].values
        y_pred = cv_df[class_name].values
        metric = dict(
            experiment=experiment_name,
            dataset=dataset_name,
            horizon=horizon,
            model=model_name,
            MAE=float(mae(y_true, y_pred)),
            MSE=float(mse(y_true, y_pred)),
            time=elapsed_time,
        )
        with open(save_dirpath / "metric.yml", "w") as f:
            yaml.dump(metric, f)

        # Release memory
        del nf
        torch.cuda.empty_cache()

# Aggregate test metrics
metrics = []
for filepath in rootpath.glob("results/**/metric.yml"):
    with open(filepath, "r") as f:
        metrics += [yaml.safe_load(f)]
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(rootpath / "metrics.csv", index=False)
print(df_metrics)

print("\nThe results were saved in", rootpath)
