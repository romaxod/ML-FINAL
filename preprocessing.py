"""Shared preprocessing for the Walmart Store Sales Forecasting project.

Both team members import from this module in every model_experiment_*.ipynb so
that all architectures are compared on exactly the same data preparation.

Contents:
  * data loading (works locally, on Kaggle and on Colab)
  * cleaning rules (markdowns, CPI/unemployment, merges)
  * holiday calendar / date features
  * WalmartFeatureBuilder - sklearn transformer used inside the final pipelines
  * series-matrix construction for the deep learning models
  * store-level totals / department-share disaggregation for classical models

NOTE: this module is shipped together with every logged MLflow model via
`code_paths=["preprocessing.py", "evaluation.py"]`, so pipelines that reference
these classes/functions can be loaded from the Model Registry anywhere.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# pandas reads .csv.zip transparently, so the Kaggle zips never need unpacking.
CANDIDATE_DIRS = [
    "data",
    "/kaggle/input/walmart-recruiting-store-sales-forecasting",
    "/content/data",
    "../final project data/walmart-recruiting-store-sales-forecasting",
    ".",
]


def find_file(fname):
    for d in CANDIDATE_DIRS:
        for suffix in ("", ".zip"):
            p = os.path.join(d, fname + suffix)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"{fname} not found - add your data folder to preprocessing.CANDIDATE_DIRS")


def load_data():
    """Return (train, test, features, stores) with parsed dates."""
    train = pd.read_csv(find_file("train.csv"), parse_dates=["Date"])
    test = pd.read_csv(find_file("test.csv"), parse_dates=["Date"])
    features = pd.read_csv(find_file("features.csv"), parse_dates=["Date"])
    stores = pd.read_csv(find_file("stores.csv"))
    return train, test, features, stores


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
MD_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def clean_features(features):
    """Cleaning rules for features.csv.

    * MarkDown1-5 do not exist before Nov 2011 -> NaN means "no promotion ran",
      so fill with 0 and keep a MarkDown_missing flag (the flag tells the model
      that the whole period is a different regime, not an average promotion).
    * CPI / Unemployment are missing for the last test months -> forward fill
      per store (macro indicators move slowly, last known value is the best guess).
    """
    f = features.copy().sort_values(["Store", "Date"])
    f["MarkDown_missing"] = f[MD_COLS].isna().all(axis=1).astype(int)
    f[MD_COLS] = f[MD_COLS].fillna(0.0)
    f["MarkDown_sum"] = f[MD_COLS].clip(lower=0).sum(axis=1)
    for c in ["CPI", "Unemployment"]:
        f[c] = f.groupby("Store")[c].transform(lambda s: s.ffill().bfill())
    return f


def merge_side(df, features, stores):
    """LEFT-join stores.csv and (cleaned) features.csv onto train/test rows."""
    out = df.merge(stores, on="Store", how="left")
    out = out.merge(features.drop(columns=["IsHoliday"]), on=["Store", "Date"],
                    how="left")
    return out


# ---------------------------------------------------------------------------
# Holiday calendar and date features
# ---------------------------------------------------------------------------
# The four holidays from the competition description; these are the w=5 weeks.
HOLIDAY_DATES = {
    "SuperBowl":    pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"]),
    "LaborDay":     pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"]),
    "Thanksgiving": pd.to_datetime(["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"]),
    "Christmas":    pd.to_datetime(["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"]),
}


def calendar_frame(dates, anchor=None):
    """Date features computed on unique dates (fast - merged onto rows afterwards).

    anchor = fixed origin for WeekIndex so train and test share one time axis.
    """
    cal = pd.DataFrame({"Date": pd.to_datetime(sorted(pd.unique(dates)))})
    anchor = pd.Timestamp(anchor) if anchor is not None else cal.Date.min()
    cal["Year"] = cal.Date.dt.year
    cal["Month"] = cal.Date.dt.month
    cal["WeekOfYear"] = cal.Date.dt.isocalendar().week.astype(int)
    cal["WeekIndex"] = ((cal.Date - anchor) // pd.Timedelta("7D")).astype(int)
    for name, hd in HOLIDAY_DATES.items():
        cal[f"Is_{name}"] = cal.Date.isin(hd).astype(int)
        # days until the next occurrence of this holiday, clipped to [0, 60]
        nxt = cal.Date.apply(
            lambda x: min([(h - x).days for h in hd if h >= x], default=999))
        cal[f"DaysTo_{name}"] = nxt.clip(0, 60)

    # How many pre-Christmas shopping days (Dec 15-24) fall into the week window
    # [Date-6, Date]. This explains why week 52 contains a different number of
    # pre-Christmas days from year to year (the key insight of the winning solution).
    def pre_xmas_days(e):
        days = pd.date_range(e - pd.Timedelta(days=6), e)
        return int(sum((d.month == 12) and (15 <= d.day <= 24) for d in days))

    cal["Pre_Christmas_Days"] = cal.Date.apply(pre_xmas_days)
    return cal


# ---------------------------------------------------------------------------
# Feature sets for the tree models
# ---------------------------------------------------------------------------
BASE_COLS = ["Store", "Dept", "Date", "IsHoliday"]

FEATURES_ALL = (
    ["Store", "Dept", "Size", "Type_A", "Type_B", "Type_C", "IsHoliday",
     "Year", "Month", "WeekOfYear", "WeekIndex",
     "Is_SuperBowl", "Is_LaborDay", "Is_Thanksgiving", "Is_Christmas",
     "DaysTo_SuperBowl", "DaysTo_LaborDay", "DaysTo_Thanksgiving", "DaysTo_Christmas",
     "Pre_Christmas_Days",
     "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    + MD_COLS + ["MarkDown_sum", "MarkDown_missing",
     "Lag_52", "Lag_52_prev", "Lag_52_next", "Lag_52_roll3", "Lag_104", "Lag52_missing",
     "SD_WOY_Mean", "SD_Mean", "SD_Median", "SD_Std", "SD_Recent13",
     "Dept_WOY_Med", "Dept_Mean", "Store_Mean", "Expected"]
)
ECON_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
MARKDOWN_COLS = MD_COLS + ["MarkDown_sum", "MarkDown_missing"]


def feature_columns(fs):
    """Resolve a feature-set name (or explicit list) to a column list."""
    if isinstance(fs, (list, tuple)):
        return list(fs)
    if fs == "all":
        return list(FEATURES_ALL)
    if fs == "no_markdown":
        return [c for c in FEATURES_ALL if c not in MARKDOWN_COLS]
    if fs == "no_econ":
        return [c for c in FEATURES_ALL if c not in ECON_COLS]
    if fs == "ts_only":
        return [c for c in FEATURES_ALL if c not in ECON_COLS + MARKDOWN_COLS]
    raise ValueError(fs)


class WalmartFeatureBuilder(BaseEstimator, TransformerMixin):
    """fit(X, y): X = raw [Store, Dept, Date, IsHoliday] rows, y = Weekly_Sales.
    transform(X): numeric feature matrix for raw rows.

    All side tables (features.csv, stores.csv) and the sales history live inside
    the fitted object, so a Pipeline(WalmartFeatureBuilder -> model) logged to
    MLflow runs directly on the raw, unprocessed test.csv.

    Lag design: the test set extends 39 weeks into the future, so only lags
    >= 52 weeks are available for every test date (direct multi-horizon
    strategy, no recursive forecasting and no error accumulation).
    """

    def __init__(self, features_df=None, stores_df=None, anchor=None,
                 feature_set="all"):
        self.features_df = features_df
        self.stores_df = stores_df
        self.anchor = anchor
        self.feature_set = feature_set

    def fit(self, X, y):
        h = X[["Store", "Dept", "Date"]].copy()
        h["Weekly_Sales"] = np.asarray(y, dtype=float)
        h["WeekOfYear"] = h.Date.dt.isocalendar().week.astype(int)
        self.hist_ = h
        g = h.groupby(["Store", "Dept"]).Weekly_Sales
        self.sd_mean_ = g.mean().rename("SD_Mean").reset_index()
        self.sd_median_ = g.median().rename("SD_Median").reset_index()
        self.sd_std_ = g.std().rename("SD_Std").reset_index()
        self.sd_woy_ = (h.groupby(["Store", "Dept", "WeekOfYear"]).Weekly_Sales
                        .mean().rename("SD_WOY_Mean").reset_index())
        self.dept_woy_ = (h.groupby(["Dept", "WeekOfYear"]).Weekly_Sales
                          .median().rename("Dept_WOY_Med").reset_index())
        self.dept_mean_ = (h.groupby("Dept").Weekly_Sales
                           .mean().rename("Dept_Mean").reset_index())
        self.store_mean_ = (h.groupby("Store").Weekly_Sales
                            .mean().rename("Store_Mean").reset_index())
        recent = h[h.Date > h.Date.max() - pd.Timedelta(weeks=13)]
        self.sd_recent_ = (recent.groupby(["Store", "Dept"]).Weekly_Sales
                           .mean().rename("SD_Recent13").reset_index())
        self.global_mean_ = float(h.Weekly_Sales.mean())
        return self

    def _lag(self, weeks, name):
        h = self.hist_[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        h["Date"] = h["Date"] + pd.Timedelta(weeks=weeks)
        return h.rename(columns={"Weekly_Sales": name})

    def transform(self, X):
        d = X.copy()
        d = d.merge(self.stores_df, on="Store", how="left")
        d = d.merge(self.features_df.drop(columns=["IsHoliday"]),
                    on=["Store", "Date"], how="left")
        d = d.merge(calendar_frame(d.Date, anchor=self.anchor), on="Date", how="left")
        for L, nm in [(52, "Lag_52"), (51, "Lag_52_prev"),
                      (53, "Lag_52_next"), (104, "Lag_104")]:
            d = d.merge(self._lag(L, nm), on=["Store", "Dept", "Date"], how="left")
        d["Lag_52_roll3"] = d[["Lag_52_prev", "Lag_52", "Lag_52_next"]].mean(axis=1)
        for t in [self.sd_woy_, self.sd_mean_, self.sd_median_, self.sd_std_,
                  self.sd_recent_, self.dept_woy_, self.dept_mean_, self.store_mean_]:
            keys = [c for c in ("Store", "Dept", "WeekOfYear") if c in t.columns]
            d = d.merge(t, on=keys, how="left")
        # fallback chain - the test set contains 11 (Store, Dept) pairs never
        # seen in train, and early history has empty lags
        d["Expected"] = (d["SD_WOY_Mean"].fillna(d["SD_Median"])
                         .fillna(d["Dept_WOY_Med"]).fillna(d["Dept_Mean"])
                         .fillna(self.global_mean_))
        d["Lag52_missing"] = d["Lag_52"].isna().astype(int)
        for c in ["Lag_52", "Lag_52_roll3", "Lag_104", "SD_WOY_Mean", "SD_Mean",
                  "SD_Median", "SD_Recent13", "Dept_WOY_Med", "Dept_Mean"]:
            d[c] = d[c].fillna(d["Expected"])
        d["Lag_52_prev"] = d["Lag_52_prev"].fillna(d["Lag_52"])
        d["Lag_52_next"] = d["Lag_52_next"].fillna(d["Lag_52"])
        d["SD_Std"] = d["SD_Std"].fillna(0.0)
        d["Store_Mean"] = d["Store_Mean"].fillna(self.global_mean_)
        d["IsHoliday"] = d["IsHoliday"].astype(int)
        for t in "ABC":
            d[f"Type_{t}"] = (d["Type"] == t).astype(int)
        return d[feature_columns(self.feature_set)].astype(float)


# Named target-transform functions (lambdas would break pickling of the pipeline)
def log1p_clip(y):
    return np.log1p(np.clip(y, 0, None))


def expm1_inv(y):
    return np.expm1(y)


# ---------------------------------------------------------------------------
# Series matrix for the deep learning models (global, direct multi-horizon)
# ---------------------------------------------------------------------------
def build_series_matrix(train_raw, horizon):
    """Pivot train into an (n_series x n_weeks) matrix on a full weekly grid.

    Decisions (logged by the notebooks in the {ARCH}_Windowing run):
      * missing weeks -> 0 (the department simply did not trade that week)
      * negative sales (returns, 1,285 rows) -> clipped to 0 for stability
      * per-series mean scaling (floor 1.0) - series levels span 4 orders of
        magnitude and an unscaled global model would only learn the big ones
      * validation cut = last `horizon` weeks (exact imitation of the test task)
    """
    all_dates = pd.date_range(train_raw.Date.min(), train_raw.Date.max(), freq="7D")
    piv = (train_raw.pivot_table(index=["Store", "Dept"], columns="Date",
                                 values="Weekly_Sales")
           .reindex(columns=all_dates))
    Y = np.nan_to_num(piv.values, nan=0.0).clip(min=0.0).astype(np.float32)
    scale = np.maximum(Y.mean(axis=1, keepdims=True), 1.0).astype(np.float32)
    hol = (train_raw.groupby("Date").IsHoliday.first()
           .reindex(all_dates).fillna(False))
    w_week = np.where(hol.values.astype(bool), 5.0, 1.0).astype(np.float32)
    T = len(all_dates)
    return SimpleNamespace(
        all_dates=all_dates, T=T, series_idx=piv.index,
        Y=Y, Ys=Y / scale, scale=scale, w_week=w_week,
        n_series=Y.shape[0],
        obs_frac=float(piv.notna().mean().mean()),
        val_cut=T - horizon, val_dates=all_dates[T - horizon:],
    )


def make_fallback_table(train_raw):
    """(Dept, WeekOfYear) median + global mean - fallback for unseen series."""
    fb = (train_raw.assign(WeekOfYear=train_raw.Date.dt.isocalendar().week.astype(int))
          .groupby(["Dept", "WeekOfYear"]).Weekly_Sales.median()
          .rename("Dept_WOY_Med").reset_index())
    return fb, float(train_raw.Weekly_Sales.mean())


# ---------------------------------------------------------------------------
# Store-level totals + department-share disaggregation (classical models)
# ---------------------------------------------------------------------------
def store_totals(tr):
    """Weekly totals per store on a full weekly grid (45 smooth series)."""
    t = tr.groupby(["Store", "Date"]).Weekly_Sales.sum().unstack("Store")
    return t.reindex(pd.date_range(t.index.min(), t.index.max(), freq="7D")).fillna(0.0)


def build_shares(tr):
    """share(Store, Dept, WeekOfYear) + overall share(Store, Dept).

    The share depends on the week of year because department mix shifts with
    the season (toys in December, garden in summer).
    """
    t = tr.assign(WOY=tr.Date.dt.isocalendar().week.astype(int))
    sd = t.groupby(["Store", "Dept", "WOY"]).Weekly_Sales.mean().rename("sd").reset_index()
    sd["tot"] = sd.groupby(["Store", "WOY"]).sd.transform("sum")
    sd["share"] = (sd.sd / sd.tot).clip(lower=0)
    ov = t.groupby(["Store", "Dept"]).Weekly_Sales.mean().rename("sd_o").reset_index()
    ov["tot_o"] = ov.groupby("Store").sd_o.transform("sum")
    ov["share_o"] = (ov.sd_o / ov.tot_o).clip(lower=0)
    return sd[["Store", "Dept", "WOY", "share"]], ov[["Store", "Dept", "share_o"]]


def disaggregate(rows, store_fc_long, shares_woy, shares_ov, global_med):
    """rows[Store, Dept, Date] + store forecast -> (Store, Dept, Date) forecast."""
    m = rows.copy()
    m["WOY"] = m.Date.dt.isocalendar().week.astype(int)
    m = m.merge(store_fc_long, on=["Store", "Date"], how="left")
    m = m.merge(shares_woy, on=["Store", "Dept", "WOY"], how="left")
    m = m.merge(shares_ov, on=["Store", "Dept"], how="left")
    share = m.share.fillna(m.share_o).fillna(0.0)
    pred = (m.store_pred * share).fillna(global_med)
    return pred.clip(lower=0).values
