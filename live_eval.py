"""
上線後的真實表現：把系統每 10 分鐘實際發出的預測記下來，30 分鐘後和真正的車數對照

這不是拿歷史資料回測，而是系統真的在運作時做的預測（和實地驗證同樣的意義，只是樣本多很多）。

紀錄：GitHub Actions 每次跑 notify.py 時呼叫 log_predictions()，
      把中大站點的預測附加到 data/pred_log_YYYY-MM-DD.csv（同一個資料時間只記一次）。
對照：python live_eval.py
      讀所有 pred_log，找出「目標時間」已經有實際資料的預測，算模型和基準（車數不變）的表現。
      python live_eval.py --md   另外把結果附加到 live_eval.md（提交後就留在 GitHub 上）
"""
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

import train

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "data", "pred_log_{}.csv")
COLS = ["資料時間", "目標時間", "站點代號", "站名", "現在可借", "預測可借", "總車柱", "模型訓練時間"]


def log_predictions(out, keywords=("中央大學",)):
    """out = train.predict() 的結果；只記站名含 keywords 的站"""
    meta = train.read_meta() or {}
    h = meta.get("horizon", train.HORIZON)
    pred_col = f"{h}分鐘後預測可借"
    d = out[out["站名"].astype(str).apply(lambda s: any(k in s for k in keywords))].copy()
    if d.empty:
        return
    t = pd.to_datetime(d["資料時間"])
    rows = pd.DataFrame({
        "資料時間": t.dt.strftime("%Y-%m-%d %H:%M"),
        "目標時間": (t + pd.Timedelta(minutes=h)).dt.strftime("%Y-%m-%d %H:%M"),
        "站點代號": d["站點代號"].astype(str),
        "站名": d["站名"].astype(str),
        "現在可借": d["現在可借"].astype(int),
        "預測可借": d[pred_col].astype(int),
        "總車柱": d["預測空位"].astype(int) + d[pred_col].astype(int),
        "模型訓練時間": str(meta.get("trained_at", ""))[:16].replace("T", " "),
    })[COLS]
    path = LOG.format(t.iloc[0].strftime("%Y-%m-%d"))
    if os.path.exists(path):       # 資料沒更新時 notify 會用同一批資料再預測一次，不要重複記
        old = pd.read_csv(path, encoding="utf-8-sig", dtype={"站點代號": str})
        seen = set(zip(old["資料時間"], old["站點代號"]))
        rows = rows[[(a, b) not in seen for a, b in zip(rows["資料時間"], rows["站點代號"])]]
    if rows.empty:
        return
    rows.to_csv(path, mode="a", index=False, header=not os.path.exists(path), encoding="utf-8-sig")
    print(f"已記錄上線預測 {len(rows)} 筆 → {os.path.basename(path)}")


def _scores(y, p, shortage):
    yc, pc = y <= shortage, p <= shortage
    tp = int((yc & pc).sum())
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "缺車判斷正確率": float((yc == pc).mean()),
        "recall": tp / yc.sum() if yc.sum() else float("nan"),
        "precision": tp / pc.sum() if pc.sum() else float("nan"),
        "漏報": int((yc & ~pc).sum()),
        "誤報": int((~yc & pc).sum()),
    }


def evaluate():
    files = sorted(glob.glob(LOG.format("*")))
    if not files:
        raise SystemExit("還沒有上線預測紀錄（data/pred_log_*.csv）。GitHub 跑過幾輪後同步再試。")
    log = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"站點代號": str}) for f in files])
    log["目標時間"] = pd.to_datetime(log["目標時間"])
    days = (log["目標時間"].max() - log["目標時間"].min()).days + 2
    actual = train.load(recent=days + 1)[["sno", "t", "sbi"]]
    actual = actual.rename(columns={"sno": "站點代號", "t": "目標時間", "sbi": "實際可借"})
    actual["實際可借"] = actual["實際可借"].round().astype(int)
    d = log.merge(actual, on=["站點代號", "目標時間"], how="inner")
    if d.empty:
        raise SystemExit("預測的目標時間都還沒到（或還沒同步到那時的資料），晚點再試。")
    shortage = (train.read_meta() or {}).get("shortage", train.SHORTAGE)
    return d, shortage


def report(d, shortage):
    y = d["實際可借"].to_numpy()
    m = _scores(y, d["預測可借"].to_numpy(), shortage)
    b = _scores(y, d["現在可借"].to_numpy(), shortage)
    lines = [
        f"## 上線後的真實表現（{datetime.now():%Y-%m-%d %H:%M} 統計）",
        "",
        f"期間 {d['資料時間'].min()} – {d['資料時間'].max()}，中大站點共 **{len(d)} 次**實際預測"
        f"（每 10 分鐘一次，30 分鐘後和真正的車數對照）；實際缺車（可借 ≤ {shortage} 台）{int((y <= shortage).sum())} 次。",
        "",
        "| | MAE（台） | 缺車判斷正確率 | 缺車 recall | 缺車 precision | 漏報 | 誤報 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in (("模型", m), ("基準（車數不變）", b)):
        lines.append(f"| {name} | {s['MAE']:.2f} | {s['缺車判斷正確率']:.1%} | {s['recall']:.2f} | "
                     f"{s['precision']:.2f} | {s['漏報']} | {s['誤報']} |")
    lines += ["", "**各站（模型／基準）**", "", "| 站名 | 次數 | MAE | 缺車 recall | 實際缺車次數 |", "|---|---|---|---|---|"]
    for sna, g in d.groupby("站名"):
        yy = g["實際可借"].to_numpy()
        sm, sb = _scores(yy, g["預測可借"].to_numpy(), shortage), _scores(yy, g["現在可借"].to_numpy(), shortage)
        rec = "—" if np.isnan(sm["recall"]) else f"{sm['recall']:.2f}／{sb['recall']:.2f}"
        lines.append(f"| {str(sna).replace('YouBike2.0_', '')} | {len(g)} | {sm['MAE']:.2f}／{sb['MAE']:.2f} | "
                     f"{rec} | {int((yy <= shortage).sum())} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="把結果附加到 live_eval.md")
    a = ap.parse_args()
    d, shortage = evaluate()
    text = report(d, shortage)
    print(text)
    if a.md:
        with open(os.path.join(HERE, "live_eval.md"), "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
        print("\n已附加到 live_eval.md")
