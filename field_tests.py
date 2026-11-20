"""
實地驗證計算（規格 F）：讀 field_tests.csv，算平均誤差與缺車預測命中率

用法：
  python field_tests.py                  # 讀專案裡的 field_tests.csv
  python field_tests.py 別的檔.csv       # 讀指定的檔案

field_tests.csv 怎麼填（用 Excel 開，存檔時選「CSV UTF-8」）：
  日期        2026-11-25            實地驗證那天
  站點        中央大學依仁堂          照 predictions.xlsx 的站名填，同一站要填一樣
  資料時間    2026-11-25 07:30      predictions.xlsx 上的「資料時間」
  目標時間    2026-11-25 08:00      資料時間 + 30 分鐘，也就是你到現場的時間
  預測可借    3                     predictions.xlsx 的「30分鐘後預測可借」
  預測缺車    是                    predictions.xlsx 的「缺車預警」，填「是」或「否」
  實際可借    1                     目標時間到現場數到的車數（或 App 截圖）
  實際缺車    是                    實際可借 ≤ 2 就是「是」（門檻見 train.py 的 SHORTAGE）
  來源        現場拍照 / App 截圖
  照片        IMG_1234.jpg          照片放組內共用雲端，這裡只記檔名
  備註        有調度車在搬

沒有預測缺車的時候也要記，預測錯的也要照實記（規格 F）。
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "field_tests.csv")

NEEDED = ["日期", "站點", "資料時間", "目標時間", "預測可借", "預測缺車", "實際可借", "實際缺車"]
YES = {"是", "y", "yes", "true", "1", "有", "缺車"}
NO = {"否", "n", "no", "false", "0", "沒有", "不缺"}


def to_bool(s):
    """把「是／否」欄位轉成 1／0；看不懂的值變成空值（之後會被略過並提醒）"""
    t = s.astype(str).str.strip().str.replace("　", "", regex=False).str.lower()
    return t.map(lambda v: 1.0 if v in YES else (0.0 if v in NO else float("nan")))


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df.columns = [str(c).strip().replace("﻿", "") for c in df.columns]
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(path)} 少了欄位：{'、'.join(missing)}\n"
                         f"目前的欄位：{'、'.join(df.columns)}")
    df["站點"] = df["站點"].astype(str).str.strip().str.replace("YouBike2.0_", "", regex=False)
    for c in ("預測可借", "實際可借"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    for c in ("預測缺車", "實際缺車"):
        df[c] = to_bool(df[c])
    return df


def stats(d):
    """算一組資料的平均誤差與缺車判斷結果"""
    err = (d["預測可借"] - d["實際可借"]).abs()
    hit = (d["預測缺車"] == d["實際缺車"])
    miss = ((d["實際缺車"] == 1) & (d["預測缺車"] == 0)).sum()      # 漏報：實際缺車但沒預警
    false_alarm = ((d["預測缺車"] == 1) & (d["實際缺車"] == 0)).sum()  # 誤報：預警了但實際不缺
    return {
        "筆數": len(d),
        "平均誤差(台)": round(err.mean(), 2),
        "最大誤差(台)": round(err.max(), 1),
        "缺車判斷正確": int(hit.sum()),
        "命中率": f"{hit.mean() * 100:.0f}%（{len(d)} 筆中 {int(hit.sum())} 筆）",
        "漏報": int(miss),
        "誤報": int(false_alarm),
        "實際缺車次數": int((d["實際缺車"] == 1).sum()),
    }


def report(path):
    df = load(path)
    print(f"資料來源：{os.path.basename(path)}")
    if df.empty:
        print("\n目前是空的（只有欄位標題）。先去實地驗證，把結果填進去再跑一次。")
        print("每天的做法：7:30 同步並打開 predictions.xlsx，記下依仁堂、圖書館的「資料時間」和")
        print("「30分鐘後預測可借」；到「資料時間 + 30 分鐘」去現場數車或 App 截圖，再填進這個檔。")
        return

    # 還沒填實際車數的先挑出來，不列入統計
    todo = df[df["實際可借"].isna() | df["實際缺車"].isna()]
    d = df.drop(todo.index)
    print(f"總筆數：{len(df)} 筆", end="")
    if len(todo):
        print(f"（其中 {len(todo)} 筆還沒填「實際可借／實際缺車」，不列入統計）")
    else:
        print()
    if d.empty:
        print("\n沒有可以統計的資料：每一筆都要填「實際可借」和「實際缺車」。")
        return

    # ===== 「預測 vs 實際」表（簡報用）=====
    t = d.copy()
    t["誤差(台)"] = (t["預測可借"] - t["實際可借"]).abs()
    t["預測缺車"] = t["預測缺車"].map({1.0: "是", 0.0: "否"})
    t["實際缺車"] = t["實際缺車"].map({1.0: "是", 0.0: "否"})
    t["判斷"] = ["正確" if p == a else ("漏報" if a == "是" else "誤報")
                 for p, a in zip(t["預測缺車"], t["實際缺車"])]
    cols = ["日期", "站點", "目標時間", "預測可借", "實際可借", "誤差(台)", "預測缺車", "實際缺車", "判斷"]
    print("\n【預測 vs 實際】")
    print(t[cols].to_string(index=False))

    # ===== 整體 =====
    s = stats(d)
    print("\n【整體】")
    print(f"  有效筆數　　　　{s['筆數']} 筆")
    print(f"  平均誤差　　　　{s['平均誤差(台)']} 台（|預測可借 − 實際可借| 的平均，最大 {s['最大誤差(台)']} 台）")
    print(f"  缺車判斷正確　　{s['缺車判斷正確']} 次，命中率 {s['命中率']}")
    print(f"  漏報（實際缺車、預測沒缺）　{s['漏報']} 次")
    print(f"  誤報（預測缺車、實際沒缺）　{s['誤報']} 次")
    print(f"  實際缺車共　　　{s['實際缺車次數']} 次")

    # ===== 依站點 =====
    rows = []
    for sta, dd in d.groupby("站點", observed=True):
        r = stats(dd)
        r["站點"] = sta
        r["天數"] = dd["日期"].astype(str).str.strip().nunique()
        rows.append(r)
    by_sta = pd.DataFrame(rows)[["站點", "天數", "筆數", "平均誤差(台)", "缺車判斷正確",
                                 "命中率", "漏報", "誤報", "實際缺車次數"]]
    print("\n【依站點】")
    print(by_sta.to_string(index=False))

    days = d["日期"].astype(str).str.strip().nunique()
    print(f"\n涵蓋 {days} 天、{d['站點'].nunique()} 站。規格 F 要求至少 3 個平日早上。")
    print("簡報寫法範例：「實地驗證 %d 天共 %d 筆，平均誤差 %s 台，缺車判斷 %s」"
          % (days, s["筆數"], s["平均誤差(台)"], s["命中率"]))
    if len(todo):
        tt = todo.copy()
        tt["預測缺車"] = tt["預測缺車"].map({1.0: "是", 0.0: "否"}).fillna("（沒填）")
        print(f"\n還沒填實際值的 {len(todo)} 筆：")
        print(tt[["日期", "站點", "目標時間", "預測可借", "預測缺車"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="實地驗證計算（規格 F）")
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="field_tests.csv 的路徑")
    report(ap.parse_args().csv)
