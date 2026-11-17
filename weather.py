"""
中大附近的雨量資料收集器（中央氣象署開放資料，自動雨量站 O-A0002-001，每 10 分鐘更新）

用法：
  python weather.py --once           # 抓一次（GitHub 每 10 分鐘會自動跑）
  python weather.py --once --debug   # 抓一次並印出測站的原始欄位（欄位對不上時用）

需要中央氣象署的授權碼：到 https://opendata.cwa.gov.tw 註冊會員 → 取得授權碼，
本機測試：PowerShell 先執行 $env:CWA_API_KEY="你的授權碼"
GitHub 上：存成 Secret，名稱 CWA_API_KEY
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"
NCU = (24.9681, 121.1946)          # 中央大學（圖書館站附近）
FIELDS = ["fetch_time", "obs_time", "station_id", "station_name", "distance_km", "rain_10m", "rain_1h"]


def find_stations(obj):
    """在任意 JSON 結構裡找出測站清單（含 StationName 的 list）"""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "StationName" in obj[0]:
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_stations(v)
            if r:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find_stations(v)
            if r:
                return r
    return None


def walk(obj, key):
    """遞迴找出第一個名為 key 的值"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = walk(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = walk(v, key)
            if r is not None:
                return r
    return None


def num(x):
    """轉數字；氣象署用負數（如 -99、-998）代表缺值或故障"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else v


def coords(st):
    cs = walk(st, "Coordinates")
    if isinstance(cs, list) and cs:
        wgs = [c for c in cs if "WGS84" in str(c.get("CoordinateName", ""))] or cs
        return num(wgs[0].get("StationLatitude")), num(wgs[0].get("StationLongitude"))
    return num(walk(st, "StationLatitude")), num(walk(st, "StationLongitude"))


def km(a, b):
    p = math.pi / 180
    h = (math.sin((b[0] - a[0]) * p / 2) ** 2
         + math.cos(a[0] * p) * math.cos(b[0] * p) * math.sin((b[1] - a[1]) * p / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def rain(st, period):
    """period: 'Past10Min' 或 'Past1hr'"""
    v = walk(st, period)
    if isinstance(v, dict):
        return num(v.get("Precipitation"))
    return num(v)


def fetch(key):
    r = requests.get(URL, params={"Authorization": key, "format": "JSON"}, timeout=30)
    r.raise_for_status()
    stations = find_stations(r.json())
    if not stations:
        raise ValueError("回傳的 JSON 找不到測站清單，請用 --debug 檢查")
    best, best_d = None, 1e9
    for st in stations:
        lat, lng = coords(st)
        if lat is None or lng is None:
            continue
        d = km(NCU, (lat, lng))
        if d < best_d and rain(st, "Past1hr") is not None:   # 只選雨量有數值的測站
            best, best_d = st, d
    if best is None:
        raise ValueError("找不到有雨量資料的測站")
    return best, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("CWA_API_KEY", "").strip()
    if not key:
        print("沒有設定 CWA_API_KEY，略過天氣資料。")
        return
    st, d = fetch(key)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "fetch_time": now,
        "obs_time": walk(st, "DateTime") or "",
        "station_id": st.get("StationId", ""),
        "station_name": st.get("StationName", ""),
        "distance_km": round(d, 2),
        "rain_10m": rain(st, "Past10Min"),
        "rain_1h": rain(st, "Past1hr"),
    }
    out = os.path.join(HERE, "data", f"weather_{now[:10]}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{now}] 測站 {row['station_name']}（距中大 {row['distance_km']} 公里）"
          f" 10 分鐘雨量 {row['rain_10m']} mm、1 小時雨量 {row['rain_1h']} mm")
    if a.debug:
        print(json.dumps(st, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("天氣資料抓取失敗：", e, file=sys.stderr)
        sys.exit(1)
