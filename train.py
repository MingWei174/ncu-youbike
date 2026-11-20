"""
YouBike 30 分鐘後可借車數預測（LightGBM）

在你電腦上（做實驗，不用改檔案）：
  python train.py                              # 正式設定訓練 → 存成 model_local.txt（不上傳）
  python train.py --horizon 60                 # 實驗：預測 60 分鐘後
  python train.py --shortage 5                 # 實驗：缺車門檻改 5 台
  python train.py --horizon 60 --shortage 10    # 兩個一起試
  python train.py --no-weather                 # 實驗：不用雨量特徵（看天氣有沒有幫助）
  python experiments.py                        # 一次跑完 v1–v5，結果寫進 experiments.md
  python train.py --predict                    # 用 GitHub 的正式模型預測 → predictions.xlsx

在 GitHub 上（自動，不用手動執行）：
  python train.py --auto     # 資料滿 7 天、且模型超過 7 天才重訓；新模型比基準好才替換
  python train.py --force    # 手動重訓（「立即重新訓練」按鈕用），一樣要比基準好才替換

決定好正式設定後，改下面的 HORIZON / SHORTAGE / CLASS_STARTS 預設值，提交並同步。

需要：pip install pandas lightgbm scikit-learn openpyxl
"""
import argparse
import glob
import json
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "snapshots*.csv")   # 會讀 data 資料夾裡所有 snapshots 檔
WEATHER = os.path.join(HERE, "data", "weather_*.csv")  # weather.py 收的雨量資料
ON_GITHUB = bool(os.environ.get("GITHUB_ACTIONS"))

# 正式模型（GitHub 訓練、會上傳）vs 本機實驗模型（不上傳）
MODEL = os.path.join(HERE, "model.txt")
META = os.path.join(HERE, "model_meta.json")
LOCAL_MODEL = os.path.join(HERE, "model_local.txt")
LOCAL_META = os.path.join(HERE, "model_meta_local.json")

# ===== 正式設定：實驗決定好之後改這裡，提交並同步 =====
HORIZON = 30            # 預測幾分鐘後
SHORTAGE = 2            # 可借車 ≤ 這個數字視為「缺車」
CLASS_STARTS = ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00"]  # 中大上課節次
# ======================================================

MIN_DAYS = 7            # 資料至少幾天才自動訓練
RETRAIN_DAYS = 7        # 正式模型超過幾天自動重訓
STEP = "10min"          # 重新取樣的時間格
NOTIFY_GAP_MIN = 60     # 同一站多久內不重複通知（分鐘）
# GitHub 上用 alert_log.json（會存回 GitHub）；本機用另一個檔，避免同步時衝突
ALERT_LOG = os.path.join(HERE, "alert_log.json" if ON_GITHUB else "alert_log_local.json")

BASE_FEATURES = ["station", "hour", "minute_of_day", "weekday", "is_weekend",
                 "mins_to_class", "tot", "sbi", "fill_ratio",
                 "sbi_lag1", "sbi_lag2", "sbi_lag3", "sbi_lag6", "sbi_diff1", "sbi_diff3"]
WEATHER_FEATURES = ["rain_10m", "rain_1h"]
FEATURES = BASE_FEATURES + WEATHER_FEATURES     # 正式模型用的特徵


def data_days():
    return len(glob.glob(DATA))


def load(recent=None):
    """recent=N：只讀最近 N 個每日檔（預測時用，比較快）"""
    files = sorted(glob.glob(DATA))
    if recent:
        files = files[-recent:]
    if not files:
        raise SystemExit("data 資料夾裡沒有 snapshots*.csv")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"sno": str}) for f in files])
    df["t"] = pd.to_datetime(df["fetch_time"])
    df = df.dropna(subset=["sbi", "tot"])
    # 每站對齊到 10 分鐘格點
    g = (df.set_index("t").groupby("sno")[["sbi", "tot"]]
           .resample(STEP).mean().groupby(level=0).ffill(limit=2).dropna().reset_index())
    names = df.drop_duplicates("sno").set_index("sno")["sna"]
    g["sna"] = g["sno"].map(names)
    return g.sort_values(["sno", "t"])


def load_weather():
    """讀雨量資料，對齊到 10 分鐘格點；沒有資料就回傳 None"""
    files = sorted(glob.glob(WEATHER))
    if not files:
        return None
    w = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files])
    w["t"] = pd.to_datetime(w["fetch_time"]).dt.floor(STEP)
    return w.groupby("t")[WEATHER_FEATURES].mean().reset_index().sort_values("t")


def add_weather(g):
    """把雨量併到每一筆站點資料（取 30 分鐘內最近一筆）；沒有雨量資料的時段留空，LightGBM 會自動處理"""
    w = load_weather()
    if w is None:
        for c in WEATHER_FEATURES:
            g[c] = np.nan
        return g
    g = g.sort_values("t")
    g = pd.merge_asof(g, w, on="t", direction="backward", tolerance=pd.Timedelta("30min"))
    return g.sort_values(["sno", "t"])


def mins_to_next_class(t):
    m = t.hour * 60 + t.minute
    starts = [int(s[:2]) * 60 + int(s[3:]) for s in CLASS_STARTS]
    future = [s - m for s in starts if s >= m]
    return min(future) if future else 999


def add_features(g, horizon_steps):
    g = g.copy()
    g["station"] = g["sno"].astype("category")
    g["hour"] = g["t"].dt.hour
    g["minute_of_day"] = g["t"].dt.hour * 60 + g["t"].dt.minute
    g["weekday"] = g["t"].dt.weekday
    g["is_weekend"] = (g["weekday"] >= 5).astype(int)
    g["mins_to_class"] = g["t"].map(mins_to_next_class)
    g["fill_ratio"] = g["sbi"] / g["tot"].replace(0, np.nan)
    by = g.groupby("sno", observed=True)["sbi"]
    for k in (1, 2, 3, 6):
        g[f"sbi_lag{k}"] = by.shift(k)
    g["sbi_diff1"] = g["sbi"] - g["sbi_lag1"]
    g["sbi_diff3"] = g["sbi"] - g["sbi_lag3"]
    g["target"] = by.shift(-horizon_steps)
    return add_weather(g)


def read_meta(path=META):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def need_auto_retrain():
    """--auto 用：回傳 (要不要訓練, 原因)"""
    days = data_days()
    if days < MIN_DAYS:
        return False, f"資料只有 {days} 天，滿 {MIN_DAYS} 天才自動訓練"
    meta = read_meta()
    if not meta or not os.path.exists(MODEL):
        return True, "還沒有正式模型"
    trained = datetime.fromisoformat(meta.get("trained_at", "2000-01-01T00:00:00"))
    age = (datetime.now() - trained).days
    if age >= RETRAIN_DAYS:
        return True, f"正式模型已經 {age} 天"
    if (meta.get("horizon") != HORIZON or meta.get("shortage") != SHORTAGE
            or meta.get("class_starts") != CLASS_STARTS or meta.get("features") != FEATURES):
        return True, "train.py 的正式設定或特徵改了"
    return False, f"正式模型才 {age} 天，還不用重訓"


def train(horizon, shortage, official, features=None, quiet=False):
    features = features or FEATURES
    steps = horizon // 10
    g = add_features(load(), steps)
    d = g.dropna(subset=["target", "sbi_lag6"])
    if len(d) < 200:
        raise SystemExit(f"資料太少（{len(d)} 筆可用），再多收幾天")

    cut = d["t"].quantile(0.8)                       # 用時間切：前 80% 訓練，後 20% 測試
    tr, te = d[d["t"] <= cut], d[d["t"] > cut]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
    m.fit(tr[features], tr["target"])

    pred = np.clip(m.predict(te[features]), 0, te["tot"])
    base = te["sbi"]                                 # 基準：假設 horizon 分鐘後跟現在一樣
    y = te["target"]
    yc = (y <= shortage).astype(int)

    def cls(p):
        pc = (p <= shortage).astype(int)
        return precision_score(yc, pc, zero_division=0), recall_score(yc, pc, zero_division=0)

    bp, br = cls(base)
    mp, mr = cls(pred)
    b_mae, m_mae = mean_absolute_error(y, base), mean_absolute_error(y, pred)
    imp = pd.Series(m.feature_importances_, features).sort_values(ascending=False)
    has_rain = any(f in features for f in WEATHER_FEATURES)
    rain_rows = int(d[WEATHER_FEATURES].notna().all(axis=1).sum()) if has_rain else 0
    meta = {"horizon": horizon, "shortage": shortage, "class_starts": CLASS_STARTS, "features": features,
            "categories": list(g["station"].cat.categories),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "data_from": f"{d['t'].min():%Y-%m-%d}", "data_to": f"{d['t'].max():%Y-%m-%d}",
            "rows_train": len(tr), "rows_test": len(te), "rows_with_rain": rain_rows,
            "top_features": list(imp.index[:5]),
            "test": {"model_mae": round(m_mae, 3), "base_mae": round(b_mae, 3),
                     "model_precision": round(mp, 3), "model_recall": round(mr, 3),
                     "base_precision": round(bp, 3), "base_recall": round(br, 3)}}
    if quiet:
        return meta

    print(f"\n訓練 {tr['t'].min():%m/%d}–{tr['t'].max():%m/%d}  {len(tr)} 筆｜測試 {te['t'].min():%m/%d}–{te['t'].max():%m/%d}  {len(te)} 筆")
    print(f"預測 {horizon} 分鐘後的可借車數，缺車門檻 ≤ {shortage} 台，{'有' if has_rain else '不用'}雨量特徵（有雨量的資料 {rain_rows} 筆）")
    print(f"{'':12}{'MAE(台)':>10}{'缺車precision':>16}{'缺車recall':>12}")
    print(f"{'基準(不變)':10}{b_mae:>10.2f}{bp:>16.2f}{br:>12.2f}")
    print(f"{'LightGBM':12}{m_mae:>10.2f}{mp:>16.2f}{mr:>12.2f}")
    print("\n最重要的特徵：", ", ".join(imp.index[:5]))

    if official:
        # 保護：新模型要贏過基準才替換正式模型
        if not (mr > br and m_mae <= b_mae):
            print("\n⚠️ 新模型沒有贏過基準（缺車 recall 要更高、MAE 不能更差），保留原本的正式模型。")
            return False
        model_path, meta_path = MODEL, META
    else:
        model_path, meta_path = LOCAL_MODEL, LOCAL_META

    # 用 Python 寫檔，避免 LightGBM 在 Windows 讀不懂中文路徑
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(m.booster_.model_to_string())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\n模型已存：{os.path.basename(model_path)}" + ("（正式模型）" if official else "（本機實驗，不會上傳）"))
    return True


def predict(write_excel=True, recent=None, update_log=True):
    """用正式模型（model.txt）預測最新一批資料"""
    meta = read_meta()
    if not meta or not os.path.exists(MODEL):
        raise SystemExit("還沒有正式模型 model.txt（GitHub 資料滿 7 天後會自動訓練，或先同步）")
    with open(MODEL, encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    h, shortage = meta["horizon"], meta.get("shortage", SHORTAGE)
    g = add_features(load(recent), h // 10)
    g["station"] = pd.Categorical(g["sno"], categories=meta["categories"])
    last = g.groupby("sno").tail(1).dropna(subset=["sbi_lag6"])
    last = last[last["t"] == last["t"].max()]       # 只預測最新一批還有資料的站
    feats = meta.get("features") or (FEATURES if booster.num_feature() == len(FEATURES) else BASE_FEATURES)
    p = np.asarray(np.clip(booster.predict(last[feats]), 0, last["tot"]).round().astype(int))
    empty = np.maximum(last["tot"].round().astype(int).to_numpy() - p, 0)   # 預測空位 = 總車柱 − 預測可借
    out = pd.DataFrame({
        "站點代號": last["sno"].values,
        "站名": last["sna"].values,
        "資料時間": last["t"].dt.strftime("%Y-%m-%d %H:%M").values,
        "現在可借": last["sbi"].astype(int).values,
        f"{h}分鐘後預測可借": p,
        "缺車預警": np.where(p <= shortage, "是", "否"),
        "預測空位": empty,
        "滿柱預警": np.where(empty <= shortage, "是", "否"),
    })
    # 同一站 NOTIFY_GAP_MIN 分鐘內只通知一次；缺車（empty）和滿柱（full）分開計算
    now = datetime.now()
    log = json.load(open(ALERT_LOG, encoding="utf-8")) if os.path.exists(ALERT_LOG) else {}
    notify, notify_full = [], []
    for sno, a_empty, a_full in zip(out["站點代號"], out["缺車預警"], out["滿柱預警"]):
        for kind, alert, result in (("empty", a_empty, notify), ("full", a_full, notify_full)):
            key = f"{sno}:{kind}"
            prev = log.get(key)
            too_soon = prev and (now - datetime.fromisoformat(prev)).total_seconds() < NOTIFY_GAP_MIN * 60
            if alert == "是" and not too_soon:
                result.append("是")
                if update_log:
                    log[key] = now.isoformat(timespec="seconds")
            else:
                result.append("否")
    out["要通知"] = notify
    out["要通知滿柱"] = notify_full
    if update_log:
        with open(ALERT_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)

    if write_excel:
        path = os.path.join(HERE, "predictions.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="預測", index=False)
            ws = w.sheets["預測"]
            from openpyxl.worksheet.table import Table
            ws.add_table(Table(displayName="Pred", ref=ws.dimensions))  # Power Automate 讀 Excel 需要「表格」
        print(f"已輸出 {os.path.basename(path)}")
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--predict", action="store_true", help="用正式模型預測，輸出 predictions.xlsx")
    a.add_argument("--horizon", type=int, default=HORIZON)
    a.add_argument("--shortage", type=int, default=SHORTAGE)
    a.add_argument("--no-weather", action="store_true", help="實驗：不用雨量特徵")
    a.add_argument("--auto", action="store_true", help="GitHub 用：條件符合才重訓正式模型")
    a.add_argument("--force", action="store_true", help="GitHub 用：立刻重訓正式模型")
    args = a.parse_args()

    if args.predict:
        print(predict(update_log=False).to_string(index=False))
    elif args.auto or args.force:
        if not ON_GITHUB:
            raise SystemExit("--auto / --force 只在 GitHub 上用。本機做實驗直接跑 python train.py")
        if args.auto:
            ok, why = need_auto_retrain()
            print("自動訓練檢查：", why)
            if not ok:
                raise SystemExit(0)
        elif data_days() < MIN_DAYS:
            raise SystemExit(f"資料只有 {data_days()} 天，滿 {MIN_DAYS} 天才能訓練正式模型")
        train(HORIZON, SHORTAGE, official=True)
    else:
        train(args.horizon, args.shortage, official=False,
              features=BASE_FEATURES if args.no_weather else FEATURES)
