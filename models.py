import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow.pyfunc
 
import preprocessing as prep
 
 
class NBeatsBlock(nn.Module):
 
    def __init__(self, L, H, width, depth, basis="generic",
                 degree=3, harmonics=6, period=52.0):
        super().__init__()
        layers, d = [], L
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.mlp = nn.Sequential(*layers)
        self.basis = basis
        if basis == "generic":
            self.theta_b = nn.Linear(width, L)
            self.theta_f = nn.Linear(width, H)
        else:
            t = np.arange(L + H, dtype=np.float32)
            if basis == "trend":
                tt = t / (L + H)
                B = np.stack([tt ** i for i in range(degree + 1)])
            else:  # seasonality - Fourier harmonics with a 52-week period
                B = np.concatenate(
                    [np.stack([np.cos(2 * np.pi * k * t / period),
                               np.sin(2 * np.pi * k * t / period)])
                     for k in range(1, harmonics + 1)]).astype(np.float32)
            self.register_buffer("Bb", torch.tensor(B[:, :L]))
            self.register_buffer("Bf", torch.tensor(B[:, L:]))
            self.theta = nn.Linear(width, B.shape[0], bias=False)
 
    def forward(self, x):
        h = self.mlp(x)
        if self.basis == "generic":
            return self.theta_b(h), self.theta_f(h)
        th = self.theta(h)
        return th @ self.Bb, th @ self.Bf
 
 
class NBeats(nn.Module):

    def __init__(self, L, H, stacks):
        super().__init__()
        blocks = []
        for basis, n_blocks, width, depth in stacks:
            for _ in range(n_blocks):
                blocks.append(NBeatsBlock(L, H, width, depth, basis))
        self.blocks = nn.ModuleList(blocks)
 
    def forward(self, x):
        residual, forecast = x, None
        for b in self.blocks:
            bc, fc = b(residual)
            residual = residual - bc
            forecast = fc if forecast is None else forecast + fc
        return forecast
 
 
def build_model(cfg, horizon):
    return NBeats(L=cfg["L"], H=horizon, stacks=cfg["stacks"])
 
 
class StoreShareForecastPipeline(mlflow.pyfunc.PythonModel):
 
    def __init__(self, store_fc_long, shares_woy, shares_ov, global_med):
        self.store_fc_long = store_fc_long
        self.shares_woy = shares_woy
        self.shares_ov = shares_ov
        self.global_med = global_med
 
    def predict(self, context, model_input):
        df = model_input.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return prep.disaggregate(df[["Store", "Dept", "Date"]], self.store_fc_long,
                                  self.shares_woy, self.shares_ov, self.global_med)