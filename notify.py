"""
自動預測 + 缺車通知（給 GitHub Actions 每 10 分鐘執行，也可以在本機測試）

流程：讀最新資料 → 用 model.txt 預測 30 分鐘後車數與空位 → 輸出 predictions.xlsx（給 Copilot Chat 小幫手）與 docs/predictions.json（給手機查詢網頁）
      → 有站預測缺車或滿柱（「要通知」或「要通知滿柱」）就呼叫 Power Automate 的 webhook

用法：
  python notify.py            # 正常執行
  python notify.py --test     # 不管有沒有缺車，送一則「測試」通知（設定時用）
  python notify.py --dry-run  # 只印出會送什麼，不真的送出

Webhook 網址從環境變數 WEBHOOK_URL 讀取（GitHub 上存在 Secrets，不寫在程式裡）。
"""
import argparse
import html
import json
import os
import sys
from datetime import datetime

import requests

import train

# 只想通知哪些站（站名關鍵字，逗號分隔）；空字串 = 全部有收集的站
WATCH = os.environ.get("WATCH_KEYWORDS", "中央大學")
# 安靜時段：這段時間外不發通知
NOTIFY_FROM, NOTIFY_TO = "06:30", "22:00"
# 最新資料超過幾分鐘就不通知（避免收集中斷時用舊資料發通知）
MAX_AGE_MIN = 30


def horizon():
    m = train.read_meta()
    return m["horizon"] if m else train.HORIZON


def shortage():
    m = train.read_meta()
    return m.get("shortage", train.SHORTAGE) if m else train.SHORTAGE


def build_payload(rows, test=False):
    title = ("【測試】" if test else "⚠️ ") + "RideWise NCU｜YouBike 缺車／滿柱預警：" + "、".join(
        f"{r['站名']}（{r['類型']}）" for r in rows)
    h = horizon()
    pred_col = f"{h}分鐘後預測可借"
    trs = "".join(
        f"<tr><td>{html.escape(str(r['站名']))}</td><td>{r['類型']}</td><td>{r['現在可借']}</td>"
        f"<td><b>{r[pred_col]}</b></td><td><b>{r['預測空位']}</b></td>"
        f"<td>{html.escape(str(r['資料時間']))}</td></tr>"
        for r in rows)
    tips = []
    if any("缺車" in r["類型"] for r in rows):
        tips.append(f"<p>缺車：預測 <b>{h} 分鐘後</b>可借車數 ≤ {shortage()} 台，建議提早出門或改去附近的站。</p>")
    if any("滿柱" in r["類型"] for r in rows):
        tips.append(f"<p>滿柱：預測 <b>{h} 分鐘後</b>空位 ≤ {shortage()} 個，建議改還附近的站。</p>")
    if any("持續" in r["類型"] for r in rows):
        tips.append(f"<p>「持續」代表這站現在就已經沒車或沒位；之後約 {train.REMIND_GAP_MIN // 60} 小時才會再提醒一次。</p>")
    body = (
        "".join(tips)
        + "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        f"<tr><th>站名</th><th>類型</th><th>現在可借</th><th>{h} 分鐘後預測可借</th>"
        f"<th>預測空位</th><th>資料時間</th></tr>{trs}</table>"
        "<p style='color:#666'>由機器學習模型自動預測，可能有誤差，請以現場為準。</p>"
    )
    if test:
        body = "<p><b>這是設定用的測試通知，數字不代表真實狀況。</b></p>" + body
    return {"title": title, "html": body, "count": len(rows),
            "stations": [str(r["站名"]) for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    url = os.environ.get("WEBHOOK_URL", "").strip()
    now = datetime.now().strftime("%H:%M")

    if a.test:
        t = datetime.now().strftime("%Y-%m-%d %H:%M")
        pc = f"{horizon()}分鐘後預測可借"
        rows = [{"站名": "（測試）中央大學依仁堂", "類型": "即將缺車", "現在可借": 5, pc: 1, "預測空位": 29, "資料時間": t},
                {"站名": "（測試）中央大學圖書館", "類型": "即將滿柱", "現在可借": 26, pc: 29, "預測空位": 1, "資料時間": t}]
    else:
        if not (os.path.exists(train.MODEL) and os.path.exists(train.META)):
            print("還沒有正式模型 model.txt（資料滿 7 天後 GitHub 會自動訓練）。略過預測。")
            return
        # 1) 不管幾點，都先更新 predictions.xlsx 和手機網頁資料
        out = train.predict(write_excel=True, recent=2, update_log=False)
        if os.environ.get("GITHUB_ACTIONS"):   # 手機網頁資料只由 GitHub 更新；失敗不影響通知
            try:
                import web
                web.write_web(out)
            except Exception as e:
                print("手機網頁資料更新失敗：", e)
            try:   # 記下這次實際發出的預測，之後用 live_eval.py 和真正的車數對照
                import live_eval
                live_eval.log_predictions(out)
            except Exception as e:
                print("上線預測紀錄失敗：", e)
        newest = datetime.strptime(out["資料時間"].max(), "%Y-%m-%d %H:%M")
        age = (datetime.now() - newest).total_seconds() / 60
        # 2) 只有在通知時段、資料夠新時才發通知
        if not (NOTIFY_FROM <= now <= NOTIFY_TO):
            print(f"現在 {now} 是安靜時段，只更新預測檔、不通知。")
            return
        if age > MAX_AGE_MIN:
            print(f"最新資料是 {age:.0f} 分鐘前，太舊了，不通知。")
            return
        out = train.predict(write_excel=False, recent=2, update_log=True)
        kws = [k for k in WATCH.split(",") if k]
        if kws:
            out = out[out["站名"].astype(str).apply(lambda s: any(k in s for k in kws))]
        print(out.to_string(index=False))
        rows = []
        for r in out.to_dict("records"):
            note = str(r.get("通知說明", ""))
            if r["要通知"] == "是":
                rows.append({**r, "類型": "持續缺車" if "持續缺車" in note else "即將缺車"})
            if r["要通知滿柱"] == "是":
                rows.append({**r, "類型": "持續滿柱" if "持續滿柱" in note else "即將滿柱"})
        if not rows:
            print("沒有需要通知的站。")
            return

    payload = build_payload(rows, test=a.test)
    print("要送出的通知：", payload["title"])
    if a.dry_run:
        print(payload["html"])
        return
    if not url:
        print("沒有設定 WEBHOOK_URL，無法送出。", file=sys.stderr)
        sys.exit(1)
    r = requests.post(url, json=payload, timeout=30)
    print("webhook 回應：", r.status_code)
    r.raise_for_status()


if __name__ == "__main__":
    main()
