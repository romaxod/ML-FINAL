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
 
 
def build_nbeats_model(cfg, horizon):
    return NBeats(L=cfg["L"], H=horizon, stacks=cfg["stacks"])


class MovingAvg(nn.Module):
    """Moving average with edge padding - the trend part of the decomposition."""

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):                      # x: (B, L)
        front = x[:, :1].repeat(1, (self.kernel - 1) // 2)
        back = x[:, -1:].repeat(1, self.kernel // 2)
        xx = torch.cat([front, x, back], dim=1).unsqueeze(1)
        return self.avg(xx).squeeze(1)


class DLinear(nn.Module):
    """Decomposition + two linear layers (trend / seasonal)."""

    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.lin_trend = nn.Linear(L, H)
        self.lin_seasonal = nn.Linear(L, H)

    def forward(self, x):
        trend = self.decomp(x)
        return self.lin_trend(trend) + self.lin_seasonal(x - trend)


class NLinear(nn.Module):
    """Last-value normalisation: x - x[-1] -> Linear -> + x[-1]."""

    def __init__(self, L, H):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, x):
        last = x[:, -1:]
        return self.lin(x - last) + last


class VanillaLinear(nn.Module):
    def __init__(self, L, H):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, x):
        return self.lin(x)


def build_dlinear_model(cfg, horizon):
    kind = cfg.get("kind", "dlinear")
    if kind == "dlinear":
        return DLinear(cfg["L"], horizon, cfg.get("kernel", 25))
    if kind == "nlinear":
        return NLinear(cfg["L"], horizon)
    return VanillaLinear(cfg["L"], horizon)


class PatchTST(nn.Module):

    def __init__(self, L, H, patch_len=8, stride=4, d_model=128, nhead=8,
                 n_layers=3, dropout=0.2, revin=True):
        super().__init__()
        self.patch_len, self.stride, self.revin = patch_len, stride, revin
        n_patches = (L - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(n_patches, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(n_patches * d_model, H))

    def forward(self, x):                       # x: (B, L)
        if self.revin:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True) + 1e-5
            x = (x - mu) / sd
        patches = x.unfold(1, self.patch_len, self.stride)   # (B, n_patches, patch_len)
        z = self.proj(patches) + self.pos
        z = self.encoder(z)
        out = self.head(z.flatten(1))
        if self.revin:
            out = out * sd + mu
        return out


def build_patchtst_model(cfg, horizon):
    return PatchTST(cfg["L"], horizon,
                    patch_len=cfg.get("patch_len", 8), stride=cfg.get("stride", 4),
                    d_model=cfg.get("d_model", 128), nhead=cfg.get("nhead", 8),
                    n_layers=cfg.get("n_layers", 3), dropout=cfg.get("dropout", 0.2),
                    revin=cfg.get("revin", True))


class TorchForecastPipeline(mlflow.pyfunc.PythonModel):
    """End-to-end pipeline: predict(raw test df [Store, Dept, Date, ...]) -> preds.
    Stores the model + last L weeks of history + scales + fallback table.
    Shared by every global-DL architecture (NBEATS, DLinear, PatchTST) - build_fn
    tells it which model class to reconstruct."""

    def __init__(self, build_fn, cfg, state_dict, hist_tail, scale_vec,
                 series_index, train_end, horizon, fallback, global_mean):
        self.build_fn = build_fn
        self.cfg = {k: v for k, v in cfg.items()}
        self.state_dict = state_dict
        self.hist_tail = hist_tail
        self.scale_vec = scale_vec
        self.pos = {sd: i for i, sd in enumerate(series_index)}
        self.train_end = train_end
        self.horizon = horizon
        self.fallback = fallback
        self.global_mean = global_mean
        self._fc = None

    def _forecast(self):
        if self._fc is None:
            import torch as _t
            model = self.build_fn(self.cfg, self.horizon)
            model.load_state_dict(self.state_dict)
            model.eval()
            with _t.no_grad():
                self._fc = (model(_t.tensor(self.hist_tail, dtype=_t.float32)).numpy()
                            * self.scale_vec).clip(min=0)
        return self._fc

    def predict(self, context, model_input):
        df = model_input.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        fc = self._forecast()
        k = (((df.Date - self.train_end).dt.days // 7) - 1).clip(0, self.horizon - 1)
        k = k.astype(int).values
        idx = np.array([self.pos.get((s, d), -1)
                        for s, d in zip(df.Store, df.Dept)])
        woy = df.Date.dt.isocalendar().week.astype(int)
        fb = (pd.DataFrame({"Dept": df.Dept.values, "WeekOfYear": woy.values})
              .merge(self.fallback, on=["Dept", "WeekOfYear"], how="left")
              ["Dept_WOY_Med"].fillna(self.global_mean).values)
        return np.where(idx >= 0, fc[idx.clip(min=0), k], fb)


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