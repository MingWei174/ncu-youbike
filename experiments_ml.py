"""
課堂用的 ML 實驗：時間序列交叉驗證 + 多個模型比較（結果附加到 experiments.md）

用法：python experiments_ml.py            # 約 5–10 分鐘
      python experiments_ml.py --folds 3  # 折數少一點，跑比較快

做什麼：
  1. 時間序列交叉驗證（sklearn TimeSeriesSplit）：依時間切成 5 折，每一折都「用過去訓練、預測之後」，
     不會偷看到未來。報告每個模型 5 折的平均 ± 標準差，比只切一次 80/20 更有說服力。
  2. 多個模型比較，全部用同一套特徵、同一個目標（預測 30 分鐘後「增減幾台」再加回現在車數），
     公平比較：基準（假設不變）、線性迴歸、隨機森林、LightGBM（正式模型的設定）。

不會動到正式模型 model.txt，也不會存任何模型；GitHub 上的系統不受影響。
"""
import argparse
import os
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

import train


def numeric(X):
    """線性迴歸、隨機森林不吃類別和缺值：站點改成代號，雨量缺值補 0"""
    X = X.copy()
    X["station"] = X["station"].cat.codes
    return X.fillna(0)


MODELS = {
    "基準（假設不變）": None,
    "線性迴歸": lambda: LinearRegression(),
    "隨機森林": lambda: RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=20,
                                          max_samples=0.3, n_jobs=-1, random_state=0),
    "LightGBM（正式設定）": lambda: lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.05,
                                                 num_leaves=31, verbose=-1),
}


def score(y, p, shortage):
    yc, pc = (y <= shortage).astype(int), (p <= shortage).astype(int)
    return (float(np.abs(p - y).mean()),
            recall_score(yc, pc, zero_division=0), precision_score(yc, pc, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    h, shortage, feats = train.HORIZON, train.SHORTAGE, train.FEATURES

    print("讀資料、做特徵 ...")
    g = train.add_features(train.load(), h // 10)
    d = g.dropna(subset=["target", "sbi_lag6"]).sort_values("t").reset_index(drop=True)
    times = np.sort(d["t"].unique())
    tss = TimeSeriesSplit(n_splits=a.folds)

    rows = []
    for k, (tr_i, te_i) in enumerate(tss.split(times), 1):
        tr = d[d["t"].isin(times[tr_i])]
        te = d[d["t"].isin(times[te_i])]
        y = te["target"].to_numpy()
        print(f"\n第 {k}/{a.folds} 折：訓練 {tr['t'].min():%m/%d %H:%M}–{tr['t'].max():%m/%d %H:%M}（{len(tr)} 筆），"
              f"測試 {te['t'].min():%m/%d %H:%M}–{te['t'].max():%m/%d %H:%M}（{len(te)} 筆）")
        for name, make in MODELS.items():
            t0 = time.time()
            if make is None:
                p = te["sbi"].to_numpy()
            else:
                m = make()
                Xtr, Xte = tr[feats], te[feats]
                if not name.startswith("LightGBM"):
                    Xtr, Xte = numeric(Xtr), numeric(Xte)
                m.fit(Xtr, tr["target"] - tr["sbi"])
                p = np.clip(te["sbi"].to_numpy() + m.predict(Xte), 0, te["tot"].to_numpy()).round()
            mae, rec, pre = score(y, p, shortage)
            rows.append({"折": k, "模型": name, "MAE": mae, "recall": rec, "precision": pre})
            print(f"   {name:<14} MAE {mae:.3f}  recall {rec:.3f}  precision {pre:.3f}  （{time.time() - t0:.0f} 秒）")

    r = pd.DataFrame(rows)
    summ = r.groupby("模型", sort=False)[["MAE", "recall", "precision"]].agg(["mean", "std"])

    def ms(name, col):
        return f"{summ.loc[name, (col, 'mean')]:.3f} ± {summ.loc[name, (col, 'std')]:.3f}"

    base = "基準（假設不變）"
    wins = r.pivot(index="折", columns="模型", values="MAE")
    lines = [f"\n## ML 實驗：時間序列交叉驗證 + 模型比較（{datetime.now():%Y-%m-%d %H:%M}）\n",
             f"資料期間 {d['t'].min():%Y-%m-%d}–{d['t'].max():%Y-%m-%d}，共 {len(d)} 筆。"
             f"預測 {h} 分鐘後，缺車門檻 ≤ {shortage} 台。`TimeSeriesSplit` {a.folds} 折：每一折都用較早的資料訓練、"
             "預測之後的時段，訓練資料一折比一折多。所有模型都預測「增減幾台」再加回現在車數，特徵相同"
             "（線性迴歸、隨機森林的站點改成代號、雨量缺值補 0）。\n",
             f"| 模型 | MAE（{a.folds} 折平均 ± 標準差） | 缺車 recall | 缺車 precision | MAE 比基準低的折數 |",
             "|---|---|---|---|---|"]
    for name in MODELS:
        n_win = "—" if name == base else f"{int((wins[name] < wins[base]).sum())}／{a.folds}"
        lines.append(f"| {name} | {ms(name, 'MAE')} | {ms(name, 'recall')} | {ms(name, 'precision')} | {n_win} |")
    lines += ["", "<details><summary>每一折的結果</summary>", "",
              "| 折 | 模型 | MAE | recall | precision |", "|---|---|---|---|---|"]
    lines += [f"| {x['折']} | {x['模型']} | {x['MAE']:.3f} | {x['recall']:.3f} | {x['precision']:.3f} |" for _, x in r.iterrows()]
    lines += ["", "</details>", "", "**解讀**：（自己填：哪個模型最好、差距大不大、標準差代表什麼）", ""]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(train.HERE, "experiments.md"), "a", encoding="utf-8") as f:
        f.write(text)
    print("已附加到 experiments.md")


if __name__ == "__main__":
    main()
