import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from os import cpu_count

import logging
import warnings

from typing import Optional

from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._scalers import TemporalNorm
from neuralforecast.common._base_auto import BaseAuto
from neuralforecast.utils import get_indexer_raise_missing

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class PatchDecomp(BaseWindows):
    """
    PatchDecomp
    """

    SAMPLING_TYPE = "windows"

    def __init__(
        self,
        h,
        input_size,
        stat_exog_list=[],
        hist_exog_list=[],
        futr_exog_list=[],
        dropout: float = 0.0,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        num_workers_loader=0,
        drop_last_loader=False,
        random_seed: int = 1,

        # Model specific hyperparameters
        time_names=[],
        embed_sizes={},
        encoder_layers: int = 3,
        hidden_size: int = 128,
        patch_size: int = 12,
        patch_stride: int = 0,
        n_head: int = 4,
        scalers_dict={},

        **trainer_kwargs
    ):
        # Inherit BaseWindows class
        super().__init__(
            h=h,
            input_size=input_size,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            **trainer_kwargs
        )
        self.example_length = input_size + h

        if patch_stride == 0:
            patch_stride = patch_size

        self.embed_sizes = embed_sizes
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.head_size = hidden_size // n_head

        self.time_names = time_names

        # Scalers
        scalers = {}
        for key in ['y'] + self.hist_exog_list + self.futr_exog_list:
            scaler_type = scalers_dict[key] if key in scalers_dict else scaler_type
            scalers[key] = TemporalNorm(
                scaler_type=scaler_type,
                dim=1,
                num_features=1,
            )
        self.scalers = nn.ModuleDict(scalers)

        # Padding
        self.input_patches = (input_size - 1) // patch_size + 1
        self.output_patches = (h - 1) // patch_size + 1
        self.input_padding = self.input_patches * patch_size - input_size
        self.output_padding = self.output_patches * patch_size - h
        self.n_patches = self.input_patches + self.output_patches
        self.patch_begins = np.arange(-self.input_patches * patch_size, self.output_patches * patch_size, patch_stride)
        self.patch_ends = self.patch_begins + patch_size
        self.t_hist = np.arange(input_size) - input_size + 1
        self.t_futr = np.arange(h) + 1
        self.t = np.concatenate([self.t_hist, self.t_futr])

        # Positional encoding
        self.z_pos = nn.Parameter(torch.Tensor(
            1, hidden_size, self.input_patches + self.output_patches))  # [1, C_hidden, (L+H)/P]
        torch.nn.init.xavier_normal_(self.z_pos)

        # Static encoder
        stat_encoders = {}
        for key in stat_exog_list:
            index = str(stat_exog_list.index(key))
            if key in embed_sizes:
                stat_encoders[index] = CategoricalEmbedding(embed_sizes[key], hidden_size)
            else:
                stat_encoders[index] = nn.Linear(1, hidden_size)
        self.stat_encoders = nn.ModuleDict(stat_encoders)

        # Time encoder
        time_encoders = {}
        for key in time_names:
            index = str(futr_exog_list.index(key))
            if key in embed_sizes:
                time_encoders[index] = CategoricalEmbedding(embed_sizes[key], hidden_size)
            else:
                time_encoders[index] = nn.Linear(1, hidden_size)
        self.time_encoders = nn.ModuleDict(time_encoders)

        # Target encoder
        self.tgt_encoder = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_stride,
        )

        # Historical encoder
        hist_encoders = {}
        for key in hist_exog_list:
            index = str(hist_exog_list.index(key))
            hist_encoders[index] = nn.Conv1d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_stride,
            )
        self.hist_encoders = nn.ModuleDict(hist_encoders)

        # Future encoder
        futr_encoders = {}
        for key in futr_exog_list:
            if key in time_names:  # Exclude time_names which used in time_encoders
                continue
            index = str(futr_exog_list.index(key))
            futr_encoders[index] = nn.Conv1d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_stride,
            )
        self.futr_encoders = nn.ModuleDict(futr_encoders)

        # Source encoders
        src_encoders = []
        for i in range(encoder_layers):
            src_encoders.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ))
        self.src_encoders = nn.ModuleList(src_encoders)

        # Patch bias encoding
        all_patches = (self.input_patches * (1 + len(self.hist_encoders))) \
            + ((self.input_patches + self.output_patches) * len(self.futr_encoders))
        self.src_skip = nn.Parameter(torch.Tensor(1, all_patches, hidden_size))  # [1, Ps, C_hidden]
        torch.nn.init.xavier_normal_(self.src_skip)

        # Target encoders
        tgt_encoders = []
        for i in range(encoder_layers):
            tgt_encoders.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ))
        self.tgt_encoders = nn.ModuleList(tgt_encoders)

        # Multi-head attentnion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_head,
            batch_first=True,
            bias=False,  # for simplicity of decomposition
        )

        # Decoder
        # https://nixtlaverse.nixtla.io/neuralforecast/examples/how_to_add_models.html#important-notes
        self.decoder = nn.Linear(
            hidden_size, patch_size * self.loss.outputsize_multiplier,
            bias=False,  # for simplicity of decomposition
        )

        # Input patch names
        patch_names = [f'y_{i}' for i in range(self.input_patches)]
        for key in hist_exog_list:
            patch_names += [f'{key}_{i}' for i in range(self.input_patches)]
        for key in futr_exog_list:
            if key in time_names:
                continue
            patch_names += [f'{key}_{i}' for i in range(self.input_patches + self.output_patches)]
        self.patch_names = patch_names

    def forward(self, windows_batch):

        # windows_batch
        y_insample = windows_batch["insample_y"][:, :, None]  # [Ws, L, 1]
        x_futr = windows_batch["futr_exog"]  # [Ws, L+H, C_futr]
        x_hist = windows_batch["hist_exog"]  # [Ws, L, C_hist]
        x_stat = windows_batch["stat_exog"]  # [Ws, C_stat]
        batch_size = y_insample.shape[0]

        # Padding
        y_insample = F.pad(y_insample.transpose(1, 2), (self.input_padding, 0))  # [Ws, 1, L]
        if x_hist is not None:
            x_hist = F.pad(x_hist.transpose(1, 2), (self.input_padding, 0))  # [Ws, C_hist, L]
        else:
            x_hist = torch.zeros([batch_size, 0, self.patch_size * self.input_patches])
        if x_futr is not None:
            x_futr = F.pad(x_futr.transpose(1, 2), (self.input_padding, self.output_padding))  # [Ws, C_futr, L+H]
        else:
            x_futr = torch.zeros([batch_size, 0, self.patch_size * (self.input_patches + self.output_patches)])

        # Encoder
        z_stat = 0
        for i, encoder in self.stat_encoders.items():
            z_stat += encoder(x_stat[:, [int(i)]])  # [Ws, 1] -> [Ws, C_hidden]
        if len(self.stat_encoders) > 0:
            z_stat = z_stat[:, :, None]  # [Ws, C_hidden] -> [Ws, C_hidden, 1]

        z_time = self.z_pos.repeat(batch_size, 1, 1)  # [Ws, C_hidden, (L+H)/P]
        for i, encoder in self.time_encoders.items():
            z_time += encoder(x_futr[:, [int(i)], self.patch_size-1::self.patch_stride].transpose(1, 2)).transpose(1, 2)  # [Ws, 1, (L+H)/P] -> [Ws, C_hidden, (L+H)/P]

        z_context = z_stat + z_time  # [Ws, C_hidden, (L+H)/P]

        z = [self.tgt_encoder(y_insample) + z_context[:, :, :self.input_patches]]  # [Ws, 1, L] -> [Ws, C_hidden, L/P]
        for i, encoder in self.hist_encoders.items():
            z += [encoder(x_hist[:, [int(i)]]) + z_context[:, :, :self.input_patches]]  # [Ws, 1, L] -> [Ws, C_hidden, L/P]
        for i, encoder in self.futr_encoders.items():
            z += [encoder(x_futr[:, [int(i)]]) + z_context]  # [Ws, 1, L+H] -> [Ws, C_hidden, (L+H)/P]
        z = torch.cat(z, dim=2)  # [Ws, C_hidden, Ps]

        src = z.transpose(1, 2)  # [Ws, Ps, C_hidden]
        tgt = z_context.transpose(1, 2)[:, self.input_patches:]  # [Ws, H/P, C_hidden]

        # Source encoders
        for encoder in self.src_encoders:
            src = src + encoder(src)  # [Ws, Ps, C_hidden]

        # Patch bias encoding
        patch_bias = torch.mul(src, self.src_skip)
        patch_bias = patch_bias.sum(dim=-1, keepdim=True)  # [Ws, Ps, 1]

        # Target encoders
        for encoder in self.tgt_encoders:
            tgt = tgt + encoder(tgt)  # [Ws, H/P, C_hidden]

        # Attention
        y, w = self.attention(query=tgt, key=src, value=src)  # [Ws, H/P, C_hidden]
        y += patch_bias.sum(dim=1, keepdim=True)  # [Ws, H/P, C_hidden]

        # Decomposition
        if self.decompose_forecast:
            self.explain(src, tgt, patch_bias)

        # Decoder
        y = self.decoder(y)  # [Ws, H/P, P*M]
        y = y.reshape(batch_size, -1, self.loss.outputsize_multiplier)[:, :self.h]  # [Ws, H, M]

        # Adapt output to loss
        y = self.loss.domain_map(y)

        return y

    def explain(
        self,
        src,
        tgt,
        patch_bias,
    ):
        """
        Decompose predicted values by patch for explanation
        https://blog.amedama.jp/entry/pytorch-multi-head-attention-verify
        """
        attention_params = {name: param.data for name, param in self.attention.named_parameters()}
        Wi = attention_params['in_proj_weight']  # [3*C_hidden, C_hidden]
        Wo = attention_params['out_proj.weight']  # [C_hidden, C_hidden]
        Wi_q, Wi_k, Wi_v = Wi.chunk(3)  # [C_hidden, C_hidden]
        avs = []
        decomp = []
        for Wq, Wk, Wv in zip(  # [C_head, C_hidden]  (C_head = C_hidden // n_head)
            Wi_q.chunk(self.n_head),
            Wi_k.chunk(self.n_head),
            Wi_v.chunk(self.n_head),
        ):
            q = torch.matmul(tgt, Wq.T)  # [Ws, H/P, C_head]
            k = torch.matmul(src, Wk.T)  # [Ws, Ps, C_head]
            v = torch.matmul(src, Wv.T)  # [Ws, Ps, C_head]
            qk = torch.bmm(q / math.sqrt(self.head_size), k.transpose(-2, -1))  # [Ws, H/P, Ps]
            a = F.softmax(qk, dim=-1)  # [Ws, H/P, Ps]
            av = torch.matmul(a, v)  # [Ws, H/P, C_head]
            avs += [av]

            v = v.unsqueeze(dim=1)  # [Ws, 1, Ps, C_head]
            a = a.unsqueeze(dim=3)  # [Ws, H/P, Ps, 1]
            d = v * a  # [Ws, H/P, Ps, C_head]
            decomp += [d]

        decomp = torch.cat(decomp, dim=-1)  # [Ws, H/P, Ps, C_hidden]
        decomp = torch.matmul(decomp, Wo.T)  # [Ws, H/P, Ps, C_hidden]
        decomp = decomp + patch_bias.unsqueeze(1)  # [Ws, H/P, Ps, C_hidden]
        decomp = torch.matmul(decomp, self.decoder.weight.T)  # [Ws, H/P, Ps, P*M]  # P:patch len, M:output dim
        decomp = decomp.transpose(1, 2)  # [Ws, Ps, H/P, P*M]
        batch_size, input_patches, output_patches, _ = decomp.shape
        decomp = decomp.reshape(batch_size, input_patches, output_patches, self.patch_size, self.loss.outputsize_multiplier)  # [Ws, Ps, H/P, P, M]
        decomp = decomp[:, :, :, :, 0]  # [Ws, Ps, H/P, P]
        scaler = self.scalers['y']
        y_scale = scaler.x_scale.reshape(batch_size, 1, 1, 1)
        y_loc = scaler.x_shift.reshape(batch_size, 1)
        decomp = decomp * y_scale  # [Ws, Ps, H/P, P]
        self.y_decomp = decomp.to('cpu').detach().numpy()
        self.y_loc = y_loc.to('cpu').detach().numpy()

        avs = torch.cat(avs, dim=-1)  # [Ws, H/P, C_hidden]
        y = torch.matmul(avs, Wo.T)  # [Ws, H/P, C_hidden]

    def _normalization(self, windows, y_idx):
        # windows are already filtered by train/validation/test
        # from the `create_windows_method` nor leakage risk
        temporal = windows["temporal"]  # B, L+H, C
        temporal_cols = windows["temporal_cols"].copy()  # B, L+H, C

        # To avoid leakage uses only the lags
        temporal_data_cols = self._get_temporal_exogenous_cols(
            temporal_cols=temporal_cols
        )
        temporal_idxs = get_indexer_raise_missing(temporal_cols, temporal_data_cols)
        temporal_idxs = np.append(y_idx, temporal_idxs)
        temporal_data = temporal[:, :, temporal_idxs]
        temporal_mask = temporal[:, :, temporal_cols.get_loc("available_mask")].clone()
        if self.h > 0:
            temporal_mask[:, -self.h :] = 0.0

        # Normalize. self.scaler stores the shift and scale for inverse transform
        temporal_mask = temporal_mask.unsqueeze(
            -1
        )  # Add channel dimension for scaler.transform.
        for i, key in enumerate(['y'] + temporal_data_cols):
            temporal_data[:, :, [i]] = self.scalers[key].transform(x=temporal_data[:, :, [i]], mask=temporal_mask)

        # Replace values in windows dict
        temporal[:, :, temporal_idxs] = temporal_data
        windows["temporal"] = temporal

        return windows

    def _inv_normalization(self, y_hat, temporal_cols, y_idx):
        # Receives window predictions [B, H, output]
        # Broadcasts outputs and inverts normalization

        # Add C dimension
        if y_hat.ndim == 2:
            remove_dimension = True
            y_hat = y_hat.unsqueeze(-1)
        else:
            remove_dimension = False

        scaler = self.scalers['y']
        y_scale = scaler.x_scale
        y_loc = scaler.x_shift

        y_scale = torch.repeat_interleave(y_scale, repeats=y_hat.shape[-1], dim=-1).to(
            y_hat.device
        )
        y_loc = torch.repeat_interleave(y_loc, repeats=y_hat.shape[-1], dim=-1).to(
            y_hat.device
        )

        y_hat = scaler.inverse_transform(z=y_hat, x_scale=y_scale, x_shift=y_loc)
        y_loc = y_loc.to(y_hat.device)
        y_scale = y_scale.to(y_hat.device)

        if remove_dimension:
            y_hat = y_hat.squeeze(-1)
            y_loc = y_loc.squeeze(-1)
            y_scale = y_scale.squeeze(-1)

        return y_hat, y_loc, y_scale


class CategoricalEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        embed_size: int,
        min_value: int = 0,
    ):
        super().__init__()
        self.emb = nn.Embedding(input_size, embed_size)
        self.min_value = min_value

    def forward(self, x=None):
        if x is not None:
            return self.emb(torch.squeeze(x, -1).to(torch.long) - self.min_value)
        return None


# class NumericalEmbedding(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         embed_size: int,
#     ):
#         super().__init__()
#         w = nn.Parameter(torch.Tensor(input_size, embed_size))
#         b = nn.Parameter(torch.zeros(input_size, embed_size))
#         torch.nn.init.xavier_normal_(w)
#         self.w = w
#         self.b = b

#     def forward(self, x=None):
#         if x is not None:
#             return torch.mul(x.unsqueeze(-1), self.w) + self.b
#         return None


class AutoPatchDecomp(BaseAuto):
    default_config = {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([32, 64, 128, 256]),
        "n_head": tune.choice([1, 2, 4, 8]),
        "encoder_layers": tune.choice([1, 2, 3, 4, 5]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
    ):
        if config is None:
            config = self.default_config.copy()
            config["input_size"] = tune.choice(
                [h * x for x in self.default_config["input_size_multiplier"]]
            )
            config["step_size"] = tune.choice([1, h])
            del config["input_size_multiplier"]
            if backend == "optuna":
                config = self._ray_config_to_optuna(config)

        super().__init__(
            cls_model=PatchDecomp,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
        )
