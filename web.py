"""
手機查詢網頁的資料：把最新預測寫成 docs/predictions.json

網頁本身是 docs/index.html（固定不變），打開時會讀這個 JSON。
GitHub Actions 每 10 分鐘跑 notify.py 時會順便呼叫 write_web()，再把 docs/ 一起推上 GitHub。
GitHub Pages 設定：Settings → Pages → Branch 選 main、資料夾選 /docs。

本機測試：python web.py   （用 GitHub 上的正式模型預測一次，寫出 docs/predictions.json）
"""
import glob
import json
import math
import os
from datetime import datetime

import pandas as pd

import train

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "predictions.json")
CAMPUS = "中央大學"      # 站名有這個關鍵字的算「校內站」
NEAR_KM = 1.5           # 校外站只列出離校內站這麼近的


def _km(lat1, lng1, lat2, lng2):
    p = math.pi / 180
    h = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lng2 - lng1) * p / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def _coords():
    """從最新一個資料檔取得各站座標"""
    files = sorted(glob.glob(os.path.join(HERE, "data", "snapshots*.csv")))
    df = pd.read_csv(files[-1], encoding="utf-8-sig", dtype={"sno": str})
    return df.drop_duplicates("sno").set_index("sno")[["lat", "lng"]]


def write_web(out):
    """out = train.predict() 回傳的表格"""
    meta = train.read_meta() or {}
    h = meta.get("horizon", train.HORIZON)
    pred_col = f"{h}分鐘後預測可借"
    xy = _coords()
    d = out.copy()
    d["站點代號"] = d["站點代號"].astype(str)
    d = d[d["站點代號"].isin(xy.index)]
    d["lat"] = d["站點代號"].map(xy["lat"])
    d["lng"] = d["站點代號"].map(xy["lng"])
    campus = d[d["站名"].astype(str).str.contains(CAMPUS)]
    if campus.empty:
        return
    d["km"] = [min(_km(la, ln, c.lat, c.lng) for c in campus.itertuples())
               for la, ln in zip(d["lat"], d["lng"])]
    d["campus"] = d["站名"].astype(str).str.contains(CAMPUS)
    d = d[d["campus"] | (d["km"] <= NEAR_KM)].sort_values(["campus", "km"], ascending=[False, True])

    stations = [{
        "name": str(r["站名"]).replace("YouBike2.0_", ""),
        "campus": bool(r["campus"]),
        "km": round(float(r["km"]), 2),
        "now": int(r["現在可借"]),
        "pred": int(r[pred_col]),
        "empty": int(r["預測空位"]) if "預測空位" in r else None,
        "shortage": r["缺車預警"] == "是",
        "full": r.get("滿柱預警") == "是",
        "time": str(r["資料時間"]),
        "lat": float(r["lat"]), "lng": float(r["lng"]),
    } for _, r in d.iterrows()]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_time": str(out["資料時間"].max()),
            "horizon": h,
            "shortage": meta.get("shortage", train.SHORTAGE),
            "model_trained": str(meta.get("trained_at", ""))[:16].replace("T", " "),
            "stations": stations,
        }, f, ensure_ascii=False, indent=1)
    print(f"已輸出手機網頁資料 docs/predictions.json（{len(stations)} 站）")


if __name__ == "__main__":
    write_web(train.predict(write_excel=False, recent=2, update_log=False))
