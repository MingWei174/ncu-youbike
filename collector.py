"""
YouBike 即時資料收集器
每隔 N 分鐘抓一次桃園 YouBike 即時資料，附加到 data/snapshots.csv。

用法：
  python collector.py --once            # 只抓一次（測試用）
  python collector.py --every 5         # 每 5 分鐘抓一次，一直跑
  python collector.py --once --area 中壢區 --keyword 中央大學,中大

預設網址來自政府資料開放平臺「桃園公共自行車即時服務資料」
(https://data.gov.tw/dataset/28228) 的 JSON 下載連結。
若網址失效，到該頁重新複製「JSON」連結貼到 API_URL。
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import requests

# ↓↓↓ 把資料平台上複製的 API 網址貼在這裡 ↓↓↓
API_URL = os.environ.get("YOUBIKE_API_URL", (
    "https://opendata.tycg.gov.tw/api/dataset/5ca2bfc7-9ace-4719-88ae-4034b9a5a55c"
    "/resource/08274d61-edbe-419d-8fcc-7a643831283d/download"
))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "snapshots.csv")
FIELDS = ["fetch_time", "sno", "sna", "sarea", "tot", "sbi", "bemp", "act", "mday", "lat", "lng"]

# 不同縣市欄位名稱不一樣，這裡統一
ALIASES = {
    "sno": ["sno", "station_no"],
    "sna": ["sna", "name_tw", "sna_tw"],
    "sarea": ["sarea", "district_tw"],
    "tot": ["tot", "tot_quantity", "total", "parking_spaces"],
    "sbi": ["sbi", "sbi_quantity", "available_rent_bikes"],
    "bemp": ["bemp", "available_return_bikes"],
    "act": ["act"],
    "mday": ["mday", "updateTime", "srcUpdateTime"],
    "lat": ["lat", "latitude"],
    "lng": ["lng", "longitude"],
}


def find_records(obj):
    """在任意 JSON 結構裡找出「站點清單」（含 sno 欄位的 list）。"""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and any(
        k in obj[0] for k in ALIASES["sno"]
    ):
        return obj
    if isinstance(obj, dict):
        # 舊版格式：{"retVal": {"0001": {...}, ...}}
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) and "sno" in v for v in vals):
            return vals
        for v in vals:
            r = find_records(v)
            if r:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find_records(v)
            if r:
                return r
    return None


def normalize(rec):
    row = {}
    for key, names in ALIASES.items():
        row[key] = next((rec[n] for n in names if n in rec), "")
    for k in ("tot", "sbi", "bemp"):
        try:
            row[k] = int(float(row[k]))
        except (TypeError, ValueError):
            row[k] = ""
    return row


def fetch(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "NCU-student-project"})
    r.raise_for_status()
    records = find_records(r.json())
    if not records:
        raise ValueError("回傳的 JSON 裡找不到站點資料，請確認 API 網址")
    return [normalize(x) for x in records]


def check_fresh(rows):
    """mday 是資料本身的更新時間，太舊代表這不是即時資料。"""
    days = []
    for r in rows:
        m = "".join(ch for ch in str(r["mday"]) if ch.isdigit())[:14]
        try:
            days.append((datetime.now() - datetime.strptime(m, "%Y%m%d%H%M%S")).total_seconds() / 3600)
        except ValueError:
            pass
    if not days:
        print("⚠️ 讀不到 mday，請人工確認資料時間")
    elif min(days) > 24:
        print(f"⚠️ 最新的 mday 是 {min(days):.0f} 小時前：這份資料不是即時的，換別的 API 網址")
    else:
        print(f"✅ 資料是新的（最新 mday 在 {min(days)*60:.0f} 分鐘前）")


def save(rows, area=None, keywords=None, daily=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = OUT.replace("snapshots.csv", f"snapshots_{now[:10]}.csv") if daily else OUT
    if area:
        rows = [r for r in rows if area in str(r["sarea"])]
    if keywords:
        rows = [r for r in rows if any(k in str(r["sna"]) for k in keywords)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"fetch_time": now, **r})
    return now, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--every", type=float, default=5, help="間隔分鐘")
    p.add_argument("--area", default=None, help="只存某行政區，例如 中壢區")
    p.add_argument("--keyword", default=None, help="站名關鍵字，逗號分隔")
    p.add_argument("--daily", action="store_true", help="每天存成一個檔案（雲端收集用）")
    a = p.parse_args()
    kws = [k for k in (a.keyword or "").split(",") if k]

    while True:
        try:
            rows = fetch(API_URL)
            t, kept = save(rows, a.area, kws, a.daily)
            print(f"[{t}] API 回傳 {len(rows)} 站，存了 {len(kept)} 站")
            if a.once:
                for r in kept[:10]:
                    print(f"  {r['sno']} {r['sna']}  可借 {r['sbi']} / 可還 {r['bemp']} / 總 {r['tot']}")
                if len(rows) == 20:
                    print("⚠️ 剛好 20 筆：網址可能只回傳部分資料（試著加上 limit 參數）")
                check_fresh(rows)
        except Exception as e:  # 收資料要能撐兩週，出錯就記錄後繼續
            print(f"[{datetime.now():%H:%M:%S}] 失敗：{e}", file=sys.stderr)
            if a.once:
                sys.exit(1)
        if a.once:
            break
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
