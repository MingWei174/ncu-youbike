"""
比較重訓前後的兩個正式模型（舊模型從 git 紀錄拿，不用自己備份）

用法（先同步，拿到 GitHub 最新的 model.txt 和資料）：
  python compare_models.py                       # 自動：比較「目前的正式模型」和「上一版正式模型」
  python compare_models.py --date 2026-12-01     # 只用某一天評估（可以給多天：--date 2026-12-01 2026-12-02）
  python compare_models.py --md                  # 結果另外附加到 compare_models.md（簡報用）

評估資料：預設用「新模型訓練資料最後一天之後」的資料，兩個模型都沒看過，比較才公平。
指標和 train.py 一樣：MAE（平均差幾台）、缺車 precision、缺車 recall；另外附上基準（假設不變）。
"""
import argparse
import json
import subprocess
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

import train


def git(*args):
    return subprocess.run(["git", *args], cwd=train.HERE, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def load_version(rev):
    """rev=None → 目前檔案；否則從 git 某個版本讀出 model.txt / model_meta.json"""
    if rev is None:
        with open(train.MODEL, encoding="utf-8") as f:
            model = f.read()
        meta = json.load(open(train.META, encoding="utf-8"))
    else:
        model = git("show", f"{rev}:model.txt")
        meta = json.loads(git("show", f"{rev}:model_meta.json"))
    return lgb.Booster(model_str=model), meta


def previous_rev():
    """最近一次改到 model.txt 的 commit，它的前一版就是舊模型"""
    revs = git("log", "--format=%h", "--", "model.txt").split()
    if len(revs) < 2:
        raise SystemExit("git 紀錄裡只有一版正式模型：還沒重訓過，或重訓的模型沒贏過基準所以沒換。")
    return revs[0] + "^"


def score(y, p, shortage):
    yc, pc = (y <= shortage).astype(int), (p <= shortage).astype(int)
    return (float(np.abs(p - y).mean()),
            precision_score(yc, pc, zero_division=0), recall_score(yc, pc, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None, help="舊模型的 git 版本（預設自動找上一版）")
    ap.add_argument("--date", nargs="*", default=None, help="評估日期 YYYY-MM-DD（預設：新模型沒看過的所有日子）")
    ap.add_argument("--keyword", default="中央大學", help="另外單獨列出的站（預設：中央大學）")
    ap.add_argument("--md", action="store_true", help="結果附加到 compare_models.md")
    a = ap.parse_args()

    old_rev = a.old or previous_rev()
    old_b, old_m = load_version(old_rev)
    new_b, new_m = load_version(None)
    if old_m.get("trained_at") == new_m.get("trained_at"):
        raise SystemExit("新舊是同一個模型，沒東西可比。")
    shortage = new_m["shortage"]

    g = train.add_features(train.load(), new_m["horizon"] // 10).dropna(subset=["sbi_lag6", "target"])
    g = g[g["t"].dt.hour >= 6]
    if a.date:
        g = g[g["t"].dt.date.astype(str).isin(a.date)]
    else:
        g = g[g["t"].dt.date.astype(str) > new_m["data_to"]]
    if g.empty:
        raise SystemExit(f"沒有可評估的資料：新模型訓練到 {new_m['data_to']}，要等之後的資料進來（或用 --date 指定）。")

    rows = []
    for label, sub in [("全部站", g), (f"{a.keyword}站", g[g["sna"].str.contains(a.keyword)])]:
        if sub.empty:
            continue
        res = {"基準（不變）": score(sub["target"].to_numpy(), sub["sbi"].to_numpy(), shortage)}
        for name, b, m in [(f"舊模型（{old_m['trained_at'][:10]}）", old_b, old_m),
                           (f"新模型（{new_m['trained_at'][:10]}）", new_b, new_m)]:
            x = sub.copy()
            x["station"] = pd.Categorical(x["sno"], categories=m["categories"])
            res[name] = score(sub["target"].to_numpy(), train.model_predict(b, m, x), shortage)
        rows.append((label, len(sub), res))

    days = sorted(g["t"].dt.date.astype(str).unique())
    lines = [f"## 新舊模型比較（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 舊模型：訓練資料 {old_m['data_from']}～{old_m['data_to']}",
             f"- 新模型：訓練資料 {new_m['data_from']}～{new_m['data_to']}",
             f"- 評估資料：{days[0]}～{days[-1]}（{len(days)} 天，06:00 之後）"
             + ("；兩個模型都沒看過" if days[0] > new_m["data_to"] else "；⚠️ 新模型訓練時看過其中部分資料，比較不公平"), ""]
    for label, n, res in rows:
        lines += [f"**{label}**（{n} 筆）", "", "| | MAE（台） | 缺車 precision | 缺車 recall |", "|---|---|---|---|"]
        lines += [f"| {k} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} |" for k, v in res.items()]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(f"{train.HERE}/compare_models.md", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print("已附加到 compare_models.md")


if __name__ == "__main__":
    main()
