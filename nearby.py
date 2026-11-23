"""
列出中央大學附近的 YouBike 站（依距離排序）

用法：
  python nearby.py            # 預設 1 公里內
  python nearby.py --km 1.5   # 改成 1.5 公里內
"""
import argparse
import glob
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CENTERS = {  # 從收集到的資料取得的座標
    "中央大學圖書館": (24.96813, 121.19467),
    "中央大學依仁堂": (24.96712, 121.19096),
    "中央大學國民運動中心": (24.96976, 121.19037),   # 屬平鎮區
}


def km(a, b):
    (lat1, lng1), (lat2, lng2) = a, b
    p = math.pi / 180
    h = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lng2 - lng1) * p / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


ap = argparse.ArgumentParser()
ap.add_argument("--km", type=float, default=1.0)
r = ap.parse_args().km

files = sorted(glob.glob(os.path.join(HERE, "data", "snapshots*.csv")))
if not files:
    raise SystemExit("data 資料夾裡沒有 snapshots*.csv，先從 GitHub 下載資料")
df = pd.read_csv(files[-1], encoding="utf-8-sig", dtype={"sno": str})
st = df.drop_duplicates("sno")[["sno", "sna", "tot", "lat", "lng"]].copy()
st["距離(公里)"] = [min(km((la, ln), c) for c in CENTERS.values()) for la, ln in zip(st["lat"], st["lng"])]
near = st[st["距離(公里)"] <= r].sort_values("距離(公里)")
near["距離(公里)"] = near["距離(公里)"].round(2)
print(near[["sno", "sna", "tot", "距離(公里)"]].rename(columns={"sno": "站點代號", "sna": "站名", "tot": "總車柱"}).to_string(index=False))
print(f"\n共 {len(near)} 站在 {r} 公里內")
