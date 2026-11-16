"""
簡報素材產生器（先跑過 python train.py 才有模型）

用法：
  python analyze.py shortage                     # 各站各時段「缺車／滿柱」時間比例 → shortage_ratio.xlsx
  python analyze.py shortage --keyword 中央大學   # 只看站名含「中央大學」的站
  python analyze.py plot --keyword 依仁堂 --date 2026-11-24
                                                 # 某站某天：實際 vs 預測 → plot_依仁堂_2026-11-24.png
  python analyze.py plot --keyword 依仁堂        # 不指定日期 → 自動挑測試期間第一個平日
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train

HERE = train.HERE
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 想看的時段：可以自己改
WINDOWS = [("早上 07:30–08:30", "07:30", "08:30"),
           ("中午 11:50–13:10", "11:50", "13:10"),
           ("下午 16:50–17:30", "16:50", "17:30"),
           ("全天 06:00–24:00", "06:00", "23:59")]


def pick(g, keyword):
    if keyword:
        g = g[g["sna"].astype(str).str.contains(keyword)]
        if g.empty:
            raise SystemExit(f"找不到站名含「{keyword}」的站")
    return g


def shortage(keyword):
    g = pick(train.load(), keyword)
    g = g[g["t"].dt.weekday < 5]                     # 只看平日
    g["bemp"] = g["tot"] - g["sbi"]
    hm = g["t"].dt.strftime("%H:%M")
    rows = []
    for (sno, sna), d in g.groupby(["sno", "sna"]):
        dh = hm.loc[d.index]
        for label, a, b in WINDOWS:
            w = d[(dh >= a) & (dh <= b)]
            if len(w) == 0:
                continue
            rows.append({
                "站點代號": sno, "站名": sna, "時段（平日）": label,
                "觀測次數": len(w),
                "借不到車比例(可借≤%d)" % train.SHORTAGE: round((w["sbi"] <= train.SHORTAGE).mean(), 3),
                "還不了車比例(空位≤%d)" % train.SHORTAGE: round((w["bemp"] <= train.SHORTAGE).mean(), 3),
                "觀測天數": w["t"].dt.date.nunique(),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("沒有平日資料可以算")
    path = os.path.join(HERE, "shortage_ratio.xlsx")
    out.to_excel(path, index=False)
    show = out.copy()
    for c in show.columns:
        if "比例" in c:
            show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(show.to_string(index=False))
    print(f"\n已輸出 {path}")
    print("簡報寫法範例：「平日早上 7:30–8:30，XX 站有 N% 的時間借不到車（資料期間 M/D–M/D，共 K 天）」")


def plot(keyword, date):
    import json
    import lightgbm as lgb
    meta = json.load(open(train.META, encoding="utf-8"))
    with open(train.MODEL, encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    h = meta["horizon"]
    g = train.add_features(train.load(), h // 10)
    g["station"] = pd.Categorical(g["sno"], categories=meta["categories"])
    g = pick(g, keyword).dropna(subset=["sbi_lag6", "target"])
    if g["sno"].nunique() > 1:
        print("符合的站：", ", ".join(g.drop_duplicates("sno")["sna"]), "→ 取第一站")
        g = g[g["sno"] == g["sno"].iloc[0]]
    if date is None:   # 預設：測試期間（後 20%）第一個平日
        cut = g["t"].quantile(0.8)
        cand = g[(g["t"] > cut) & (g["t"].dt.weekday < 5)]["t"].dt.date
        if cand.empty:
            raise SystemExit("測試期間沒有平日，請用 --date 指定")
        date = str(cand.iloc[0])
    d = g[g["t"].dt.date.astype(str) == date].copy()
    d = d[(d["t"].dt.hour >= 6)]
    if d.empty:
        raise SystemExit(f"{date} 沒有資料")
    d["pred"] = np.clip(booster.predict(d[train.FEATURES]), 0, d["tot"])
    d["when"] = d["t"] + pd.Timedelta(minutes=h)     # 預測是針對 h 分鐘後
    name = str(d["sna"].iloc[0]).replace("YouBike2.0_", "")

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    ax.plot(d["when"], d["target"], color="#1f2937", lw=2.2, label="實際可借車數")
    ax.plot(d["when"], d["pred"], color="#2563eb", lw=2, ls="--", label=f"{h} 分鐘前的模型預測")
    ax.plot(d["when"], d["sbi"], color="#9ca3af", lw=1.2, ls=":", label=f"基準：{h} 分鐘前的即時車數（假設不變）")
    ax.axhspan(0, train.SHORTAGE, color="#fee2e2", label=f"缺車區（≤{train.SHORTAGE} 台）")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{name}　{date}（{'一二三四五六日'[pd.Timestamp(date).weekday()]}）　實際 vs 預測", fontsize=14)
    ax.set_ylabel("可借車數（台）")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = os.path.join(HERE, f"plot_{keyword or 'station'}_{date}.png")
    fig.savefig(path)
    mae_m = (d["pred"] - d["target"]).abs().mean()
    mae_b = (d["sbi"] - d["target"]).abs().mean()
    print(f"{name} {date}：模型平均誤差 {mae_m:.2f} 台，基準 {mae_b:.2f} 台")
    print(f"已輸出 {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["shortage", "plot"])
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    a = ap.parse_args()
    shortage(a.keyword) if a.what == "shortage" else plot(a.keyword, a.date)
