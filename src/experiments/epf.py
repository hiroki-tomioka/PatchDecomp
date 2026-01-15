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
from neuralforecast.models import NBEATSx, NHITS, TFT, PatchTST
from neuralforecast.auto import (
    AutoNBEATS,
    AutoNBEATSx,
    AutoNHITS,
    AutoTFT,
    AutoTSMixerx,
    AutoTiDE,
)
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mae, mse
from ray import tune

sys.path.append("..")
from models.patchdecomp import AutoPatchDecomp
from utils import ray2optuna, load_epf_dataset


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.backends.cudnn.deterministic = True


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--unique_ids",
    nargs="*",
    type=str,
    choices=["PJM", "NP", "BE", "FR", "DE"],
    default=["BE"],
    help="You can designate single market (PJM, NP, BE, FR, or DE) or all the markets at the same time.",
)
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# Load datasets
df, static_df = load_epf_dataset(args.unique_ids)

# Train/val/test split (based on TimeXer's experimental setup)
val_size = 5448
test_size = 10464

# Setting
now = dt.datetime.now().strftime("%y%m%d_%H%M%S")
_unique_ids = "_".join(args.unique_ids)
rootpath = pathlib.Path(f"../../results/epf/{_unique_ids}/seed{args.seed}_{now}/")
horizon = 24
input_size = 7 * horizon
step_size = 1
futr_exog_list = ["Exogenous 1", "Exogenous 2", "month", "week_day", "hour"]
hist_exog_list = []
if len(args.unique_ids) > 1:
    stat_exog_list = ["market_id"]
    embed_sizes = {"market_id": 5}
else:
    stat_exog_list = []
    embed_sizes = {}
max_epochs = 2000
early_stop_patience_steps = 20
n_trials = 200
scaler_type = "standard"
batch_size = 16
windows_batch_size = 512
learning_rate = 1e-3
random_seed = args.seed
backend = "optuna"

# Model configs
config_models = dict(
    PatchDecomp=dict(
        class_name="AutoPatchDecomp",
        patch_size=horizon,
        hidden_size=tune.choice([16, 32, 64, 128, 256, 512]),
        n_head=tune.choice([4, 8]),
        encoder_layers=tune.choice([1, 2, 3, 4]),
        dropout=tune.uniform(0.0, 0.5),
        time_names=[],
        scalers_dict={
            "week_day": "identity",
            "month": "identity",
            "hour": "identity",
        },
        embed_sizes=embed_sizes,
    ),
    NBEATS=dict(
        class_name="AutoNBEATSx",
        n_harmonics=2,
        n_polynomials=2,
        stack_types=["identity", "trend", "seasonality"],
        n_blocks=[1, 1, 1],
        mlp_units=tune.choice(
            [
                3 * [[16, 16]],
                3 * [[32, 32]],
                3 * [[64, 64]],
                3 * [[128, 128]],
                3 * [[256, 256]],
                3 * [[512, 512]],
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
                3 * [[16, 16]],
                3 * [[32, 32]],
                3 * [[64, 64]],
                3 * [[128, 128]],
                3 * [[256, 256]],
                3 * [[512, 512]],
            ]
        ),
        dropout_prob_theta=tune.uniform(0.0, 0.5),
    ),
    TFT=dict(
        class_name="AutoTFT",
        hidden_size=tune.choice([16, 32, 64, 128, 256, 512]),
        n_head=tune.choice([4, 8]),
        dropout=tune.uniform(0.0, 0.5),
    ),
    TSMixer=dict(
        class_name="AutoTSMixerx",
        n_series=1,
        n_block=tune.choice([1, 2, 4, 6, 8]),
        ff_dim=tune.choice([16, 32, 64, 128, 256, 512]),
        dropout=tune.uniform(0.0, 0.5),
    ),
    TiDE=dict(
        class_name="AutoTiDE",
        hidden_size=tune.choice([16, 32, 64, 128, 256, 512]),
        decoder_output_dim=tune.choice([16, 32, 64, 128, 256, 512]),
        temporal_decoder_dim=tune.choice([16, 32, 64, 128, 256, 512]),
        num_encoder_layers=tune.choice([1, 2, 3, 4]),
        num_decoder_layers=tune.choice([1, 2, 3, 4]),
        dropout=tune.uniform(0.0, 0.5),
    ),
)

# Main loop
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
        futr_exog_list=futr_exog_list,
        hist_exog_list=hist_exog_list,
        stat_exog_list=stat_exog_list,
        scaler_type=scaler_type,
        batch_size=batch_size,
        logger=[
            pl.loggers.TensorBoardLogger(
                save_dir=rootpath,
                name="logs",
                version=model_name,
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
        loss=MAE(),
        config=config,
        search_alg=optuna.samplers.TPESampler(seed=random_seed),
        num_samples=n_trials,
        backend=backend,
    )
    if "AutoTSMixer" in class_name:  # n_series is a required argument
        model_kws["n_series"] = n_series
    model = eval(class_name)(**model_kws)
    nf = NeuralForecast(
        models=[model],
        freq="H",
    )

    # Train, valid, test
    t0 = time.time()
    try:
        cv_df = nf.cross_validation(
            df=df,
            static_df=static_df,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
            n_windows=None,
        )
    except Exception as e:
        print("#" * 64)
        print("Fail")
        print(class_name)
        print(e)
        continue
    elapsed_time = time.time() - t0
    print("#" * 64)
    print("Success")
    print(class_name)
    print(cv_df)

    # Make directory
    save_dirpath = rootpath / "results" / model_name
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
        model=model_name,
        MAE=float(mae(y_true, y_pred)),
        MSE=float(mse(y_true, y_pred)),
        time=elapsed_time,
    )
    with open(save_dirpath / "metric.yml", "w") as f:
        yaml.dump(metric, f)
    if len(args.unique_ids) > 1:
        metrics = []
        for unique_id in args.unique_ids:
            y_true = cv_df.loc[unique_id, "y"].values
            y_pred = cv_df.loc[unique_id, class_name].values
            metrics.append(
                dict(
                    model=model_name,
                    market=unique_id,
                    MAE=float(mae(y_true, y_pred)),
                    MSE=float(mse(y_true, y_pred)),
                )
            )
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(save_dirpath / "metrics_per_market.csv", index=False)
        print(df_metrics)

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

# Aggregate test metrics (per market)
if len(args.unique_ids) > 1:
    metrics = []
    for filepath in rootpath.glob("results/**/metrics_per_market.csv"):
        metrics.append(pd.read_csv(filepath))
    df_metrics = pd.concat(metrics)
    df_metrics.to_csv(rootpath / "metrics_per_market.csv", index=False)
    print(df_metrics)

print("\nThe results were saved in", rootpath)
