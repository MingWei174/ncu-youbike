"""
產生「假的」測試資料，只用來確認 train.py 跑得動。
⚠️ 這份資料的數字不能放進簡報！真正的結果要用 collector.py 收到的資料。

用法：python make_fake_data.py   → 產生 data/snapshots.csv（若已有真資料會拒絕覆蓋）
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "snapshots.csv")
import glob
if glob.glob(os.path.join(HERE, "data", "snapshots*.csv")):
    raise SystemExit("data 資料夾裡已經有資料，為了不跟真資料混在一起，請在另一個資料夾測試")

rng = np.random.default_rng(0)
stations = [("T001", "測試站A(宿舍)", 30), ("T002", "測試站B(校門)", 40), ("T003", "測試站C(圖書館)", 20)]
start = datetime(2026, 11, 2)
rows = []
for sno, sna, tot in stations:
    sbi = tot / 2
    for i in range(14 * 24 * 12):                      # 14 天，每 5 分鐘
        t = start + timedelta(minutes=5 * i)
        h = t.hour + t.minute / 60
        weekday = t.weekday() < 5
        # 早上 7:30–8:10 宿舍站被借光、校門站滿柱（只是模擬）
        rush = weekday and 7.5 <= h <= 8.2
        drift = (-2.5 if sno == "T001" else 2.0 if sno == "T002" else 0) if rush else (tot / 2 - sbi) * 0.05
        sbi = float(np.clip(sbi + drift + rng.normal(0, 0.8), 0, tot))
        rows.append([t.strftime("%Y-%m-%d %H:%M:%S"), sno, sna, "測試區", tot, round(sbi), tot - round(sbi), 1, "", 0, 0])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows, columns=["fetch_time", "sno", "sna", "sarea", "tot", "sbi", "bemp", "act", "mday", "lat", "lng"]) \
  .to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"已產生假資料 {len(rows)} 筆 → {OUT}")
