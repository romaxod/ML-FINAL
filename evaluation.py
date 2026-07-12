import numpy as np
import pandas as pd


def wmae(y_true, y_pred, is_holiday):
    w = np.where(np.asarray(is_holiday).astype(bool), 5.0, 1.0)
    return float(np.sum(w * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(w))


def holdout_split(train_raw, val_weeks=13):
    cutoff = train_raw.Date.max() - pd.Timedelta(weeks=val_weeks)
    train_part = train_raw[train_raw.Date <= cutoff].copy()
    val_part = train_raw[train_raw.Date > cutoff].copy()
    return train_part, val_part, cutoff


def expanding_folds(train_raw, n_folds=3, fold_weeks=13):
    end = train_raw.Date.max()
    for k in range(n_folds):
        hi = end - pd.Timedelta(weeks=fold_weeks * k)
        lo = hi - pd.Timedelta(weeks=fold_weeks)
        tr = train_raw[train_raw.Date <= lo]
        vl = train_raw[(train_raw.Date > lo) & (train_raw.Date <= hi)]
        yield k, tr, vl, lo, hi


def make_submission(test_df, preds, path):
    sub = pd.DataFrame({
        "Id": (test_df.Store.astype(str) + "_" + test_df.Dept.astype(str) + "_"
               + test_df.Date.dt.strftime("%Y-%m-%d")),
        "Weekly_Sales": np.asarray(preds),
    })
    sub.to_csv(path, index=False)
    return sub


def apply_christmas_shift(sub_df, test_df, r=2.5 / 7):
    s = sub_df.copy()
    t = test_df.copy()
    t["woy"] = t.Date.dt.isocalendar().week.astype(int)
    t["pred"] = s.Weekly_Sales.values
    piv = t.pivot_table(index=["Store", "Dept"], columns="woy", values="pred")
    adj = piv.copy()
    for w in [49, 50, 51, 52]:
        if w in piv.columns and (w - 1) in piv.columns:
            adj[w] = (1 - r) * piv[w] + r * piv[w - 1]
    lookup = adj.stack().rename("pred_adj").reset_index()
    t = t.merge(lookup, on=["Store", "Dept", "woy"], how="left")
    s["Weekly_Sales"] = t.pred_adj.fillna(t.pred).values
    return s
