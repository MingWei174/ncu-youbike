"""
簡報素材產生器（先跑過 python train.py 才有模型）

用法：
  python analyze.py shortage                     # 各站各時段「缺車／滿柱」時間比例 → shortage_ratio.xlsx
  python analyze.py shortage --keyword 中央大學   # 只看站名含「中央大學」的站
  python analyze.py usage --keyword 中央大學      # 使用量分析：每站每天估計借出台數 + 熱力圖 → usage_中央大學.png
  python analyze.py usage --keyword 中央大學 --weekday-only   # 只看平日
  python analyze.py plot --keyword 依仁堂 --date 2026-11-24
                                                 # 某站某天：實際 vs 預測 → plot_依仁堂_2026-11-24.png
  python analyze.py plot --keyword 依仁堂        # 不指定日期 → 自動挑測試期間第一個平日
  python analyze.py plot --keyword 依仁堂 --official   # 畫 GitHub 的正式模型
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

# 想看的時段
WINDOWS = [("早上 07:30–09:30", "07:30", "09:30"),
           ("中午 11:50–13:30", "11:50", "13:30"),
           ("下午 16:50–17:50", "16:50", "17:50"),
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


WEEKDAY_NAMES = ["一", "二", "三", "四", "五", "六", "日"]
ESTIMATE_NOTE = "估計值（下限）：同一站相鄰兩筆（10 分鐘）車數變少的台數算「至少被借走」，變多的算「至少被還回」；10 分鐘內同時有借有還會互相抵消，所以是低估。"


def borrow_return(g):
    """算每一筆相對「前一筆（10 分鐘前）」的估計借出／還回台數。

    只計算真的相鄰 10 分鐘的兩筆（load() 補值失敗會留下時間空隙，
    跨空隙的差值不是 10 分鐘內的變化，不能算進來）。
    """
    g = g.sort_values(["sno", "t"]).copy()
    by = g.groupby("sno", observed=True)
    gap = by["t"].diff().dt.total_seconds().div(60)   # 與前一筆相隔幾分鐘
    d = by["sbi"].diff()                             # 可借車數變化
    ok = gap.eq(10)
    g["估計借出"] = np.where(ok & d.lt(0), -d, 0.0)   # 變少 → 至少被借走
    g["估計還回"] = np.where(ok & d.gt(0), d, 0.0)    # 變多 → 至少被還回
    g["有效配對"] = ok.astype(int)
    g["日期"] = g["t"].dt.date
    g["weekday"] = g["t"].dt.day_of_week
    g["hour"] = g["t"].dt.hour
    return g


def usage(keyword, weekday_only=False):
    """使用量分析：從車數變化估算每站每天被借走幾台，並畫「星期 × 小時」熱力圖"""
    g = borrow_return(pick(train.load(), keyword))
    if weekday_only:
        g = g[g["weekday"] < 5]
        if g.empty:
            raise SystemExit("沒有平日資料可以算")

    # ===== 每站每天的估計借出／還回 =====
    per_day = (g.groupby(["sno", "sna", "日期"], observed=True)
                .agg(估計借出=("估計借出", "sum"), 估計還回=("估計還回", "sum"),
                     有效配對=("有效配對", "sum"), 觀測筆數=("t", "size"))
                .reset_index())
    per_day["站名"] = per_day["sna"].astype(str).str.replace("YouBike2.0_", "", regex=False)
    per_day["星期"] = pd.to_datetime(per_day["日期"]).dt.day_of_week.map(lambda w: WEEKDAY_NAMES[w])
    for c in ("估計借出", "估計還回"):
        per_day[c] = per_day[c].round(1)
    show = per_day.rename(columns={"sno": "站點代號"})[
        ["站點代號", "站名", "日期", "星期", "估計借出", "估計還回", "有效配對", "觀測筆數"]]
    print(show.to_string(index=False))

    # 每站總計（整個資料期間）
    tot = (per_day.groupby(["站名"], observed=True)
                  .agg(天數=("日期", "nunique"), 估計借出=("估計借出", "sum"), 估計還回=("估計還回", "sum"))
                  .reset_index())
    tot["每天平均借出"] = (tot["估計借出"] / tot["天數"]).round(1)
    print("\n各站總計（資料期間 %s–%s%s）" % (g["t"].min().strftime("%m/%d"), g["t"].max().strftime("%m/%d"),
                                        "，只算平日" if weekday_only else ""))
    print(tot.to_string(index=False))

    # ===== 星期 × 小時 平均借出台數熱力圖 =====
    cell = g.groupby(["weekday", "hour"], observed=True).agg(s=("估計借出", "sum"), n=("日期", "nunique"))
    avg = (cell["s"] / cell["n"]).unstack("hour").reindex(index=range(7), columns=range(24))
    rows_wd = sorted(g["weekday"].unique())          # 只畫有資料的星期
    avg = avg.loc[rows_wd]

    name = keyword or "全部站點"
    n_sta = g["sno"].nunique()
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("#f1f3f5")                           # 沒資料的格子留白
    fig, ax = plt.subplots(figsize=(12.5, 1.1 + 0.52 * len(rows_wd)), dpi=150)
    im = ax.imshow(np.ma.masked_invalid(avg.values), aspect="auto", cmap=cmap)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(rows_wd)))
    ax.set_yticklabels([f"星期{WEEKDAY_NAMES[w]}" for w in rows_wd], fontsize=9)
    ax.set_xlabel("小時", fontsize=9)
    vmax = np.nanmax(avg.values) if np.isfinite(avg.values).any() else 0
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            v = avg.values[i, j]
            if np.isfinite(v) and v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if vmax and v > vmax * 0.6 else "#1f2937")
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows_wd), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{name}　平均每小時估計借出台數（星期 × 小時）\n"
                 f"{n_sta} 站・資料期間 {g['t'].min():%Y/%m/%d}–{g['t'].max():%m/%d}"
                 f"{'・只算平日' if weekday_only else ''}・估計值（下限）", fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("平均借出台數／小時（估計值，下限）", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(HERE, f"usage_{name}.png")
    fig.savefig(path)

    print("\n" + ESTIMATE_NOTE)
    print(f"已輸出 {path}（圖片不用提交）")
    if avg.notna().any().any():
        flat = avg.stack()
        top = flat.sort_values(ascending=False).head(3)
        hot = "、".join(f"星期{WEEKDAY_NAMES[w]} {h:02d}:00（約 {v:.0f} 台）" for (w, h), v in top.items())
        print(f"借車最多的時段：{hot}")
    print("簡報圖說範例：「%s 使用量熱力圖（%s–%s%s），數字為估計值（下限）」"
          % (name, g["t"].min().strftime("%m/%d"), g["t"].max().strftime("%m/%d"),
             "，平日" if weekday_only else ""))


def plot(keyword, date, use_official=False):
    import json
    import lightgbm as lgb
    # 有本機實驗模型就畫它，沒有就畫 GitHub 的正式模型
    if os.path.exists(train.LOCAL_MODEL) and not use_official:
        mpath, metapath, which = train.LOCAL_MODEL, train.LOCAL_META, "本機實驗模型"
    else:
        mpath, metapath, which = train.MODEL, train.META, "正式模型"
    if not os.path.exists(mpath):
        raise SystemExit("找不到模型：先在本機跑 python train.py，或同步拿到 GitHub 的 model.txt")
    print("使用：", which)
    meta = json.load(open(metapath, encoding="utf-8"))
    with open(mpath, encoding="utf-8") as f:
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
    d["pred"] = train.model_predict(booster, meta, d)   # 和 predict() 同一套算法
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
    ap.add_argument("what", choices=["shortage", "usage", "plot"])
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--official", action="store_true", help="畫 GitHub 的正式模型（預設：有本機實驗模型就畫它）")
    ap.add_argument("--weekday-only", action="store_true", help="usage 用：只算平日")
    a = ap.parse_args()
    if a.what == "shortage":
        shortage(a.keyword)
    elif a.what == "usage":
        usage(a.keyword, a.weekday_only)
    else:
        plot(a.keyword, a.date, a.official)
