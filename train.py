"""
YouBike 30 分鐘後可借車數預測（LightGBM）

用法：
  python train.py                 # 用 data/snapshots.csv 訓練 + 評估，模型存到 model.txt
  python train.py --predict       # 用最新一筆資料預測，輸出 predictions.xlsx
  python train.py --horizon 60    # 改成預測 60 分鐘後

需要：pip install pandas lightgbm scikit-learn openpyxl
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "snapshots*.csv")   # 會讀 data 資料夾裡所有 snapshots 檔
MODEL = os.path.join(HERE, "model.txt")
META = os.path.join(HERE, "model_meta.json")
STEP = "10min"          # 重新取樣的時間格
SHORTAGE = 2            # 可借車 ≤ 2 視為「缺車」

# 上課開始時間：請改成中大實際的節次時間
CLASS_STARTS = ["08:00", "09:00", "10:10", "11:10", "13:00", "14:00", "15:10", "16:10"]

FEATURES = ["station", "hour", "minute_of_day", "weekday", "is_weekend",
            "mins_to_class", "tot", "sbi", "fill_ratio",
            "sbi_lag1", "sbi_lag2", "sbi_lag3", "sbi_lag6", "sbi_diff1", "sbi_diff3"]


def load():
    files = sorted(glob.glob(DATA))
    if not files:
        raise SystemExit("data 資料夾裡沒有 snapshots*.csv")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"sno": str}) for f in files])
    df["t"] = pd.to_datetime(df["fetch_time"])
    df = df.dropna(subset=["sbi", "tot"])
    # 每站對齊到 10 分鐘格點
    g = (df.set_index("t").groupby("sno")[["sbi", "tot"]]
           .resample(STEP).mean().groupby(level=0).ffill(limit=2).dropna().reset_index())
    names = df.drop_duplicates("sno").set_index("sno")["sna"]
    g["sna"] = g["sno"].map(names)
    return g.sort_values(["sno", "t"])


def mins_to_next_class(t):
    m = t.hour * 60 + t.minute
    starts = [int(s[:2]) * 60 + int(s[3:]) for s in CLASS_STARTS]
    future = [s - m for s in starts if s >= m]
    return min(future) if future else 999


def add_features(g, horizon_steps):
    g = g.copy()
    g["station"] = g["sno"].astype("category")
    g["hour"] = g["t"].dt.hour
    g["minute_of_day"] = g["t"].dt.hour * 60 + g["t"].dt.minute
    g["weekday"] = g["t"].dt.weekday
    g["is_weekend"] = (g["weekday"] >= 5).astype(int)
    g["mins_to_class"] = g["t"].map(mins_to_next_class)
    g["fill_ratio"] = g["sbi"] / g["tot"].replace(0, np.nan)
    by = g.groupby("sno", observed=True)["sbi"]
    for k in (1, 2, 3, 6):
        g[f"sbi_lag{k}"] = by.shift(k)
    g["sbi_diff1"] = g["sbi"] - g["sbi_lag1"]
    g["sbi_diff3"] = g["sbi"] - g["sbi_lag3"]
    g["target"] = by.shift(-horizon_steps)
    return g


def train(horizon):
    steps = horizon // 10
    g = add_features(load(), steps)
    d = g.dropna(subset=["target", "sbi_lag6"])
    if len(d) < 200:
        raise SystemExit(f"資料太少（{len(d)} 筆可用），再多收幾天")

    cut = d["t"].quantile(0.8)                       # 用時間切：前 80% 訓練，後 20% 測試
    tr, te = d[d["t"] <= cut], d[d["t"] > cut]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
    m.fit(tr[FEATURES], tr["target"])

    pred = np.clip(m.predict(te[FEATURES]), 0, te["tot"])
    base = te["sbi"]                                 # 基準：假設 30 分鐘後跟現在一樣
    y = te["target"]
    yc = (y <= SHORTAGE).astype(int)

    def cls(p):
        pc = (p <= SHORTAGE).astype(int)
        return precision_score(yc, pc, zero_division=0), recall_score(yc, pc, zero_division=0)

    bp, br = cls(base)
    mp, mr = cls(pred)
    print(f"\n訓練 {tr['t'].min():%m/%d}–{tr['t'].max():%m/%d}  {len(tr)} 筆｜測試 {te['t'].min():%m/%d}–{te['t'].max():%m/%d}  {len(te)} 筆")
    print(f"預測 {horizon} 分鐘後的可借車數")
    print(f"{'':12}{'MAE(台)':>10}{'缺車precision':>16}{'缺車recall':>12}")
    print(f"{'基準(不變)':10}{mean_absolute_error(y, base):>10.2f}{bp:>16.2f}{br:>12.2f}")
    print(f"{'LightGBM':12}{mean_absolute_error(y, pred):>10.2f}{mp:>16.2f}{mr:>12.2f}")
    imp = pd.Series(m.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\n最重要的特徵：", ", ".join(imp.index[:5]))

    m.booster_.save_model(MODEL)
    json.dump({"horizon": horizon, "categories": list(g["station"].cat.categories)},
              open(META, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n模型已存：{MODEL}")


def predict():
    meta = json.load(open(META, encoding="utf-8"))
    booster = lgb.Booster(model_file=MODEL)
    g = add_features(load(), meta["horizon"] // 10)
    g["station"] = pd.Categorical(g["sno"], categories=meta["categories"])
    last = g.groupby("sno").tail(1).dropna(subset=["sbi_lag6"])
    p = np.clip(booster.predict(last[FEATURES]), 0, last["tot"]).round().astype(int)
    out = pd.DataFrame({
        "站點代號": last["sno"].values,
        "站名": last["sna"].values,
        "資料時間": last["t"].dt.strftime("%Y-%m-%d %H:%M").values,
        "現在可借": last["sbi"].astype(int).values,
        f"{meta['horizon']}分鐘後預測可借": p,
        "缺車預警": np.where(p <= SHORTAGE, "是", "否"),
    })
    path = os.path.join(HERE, "predictions.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="預測", index=False)
        ws = w.sheets["預測"]
        from openpyxl.worksheet.table import Table
        ws.add_table(Table(displayName="Pred", ref=ws.dimensions))  # Power Automate 讀 Excel 需要「表格」
    print(out.to_string(index=False))
    print(f"\n已輸出 {path}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--predict", action="store_true")
    a.add_argument("--horizon", type=int, default=30)
    args = a.parse_args()
    predict() if args.predict else train(args.horizon)
