"""
一次跑完所有實驗，結果自動寫進 experiments.md 和 experiments.csv

用法：python experiments.py
跑完後在 VS Code 提交並同步，結果就會在 GitHub 上（不需要截圖）。
不會動到正式模型 model.txt，也不會存本機模型。
"""
import csv
import os
from datetime import datetime

import train

EXPERIMENTS = [
    ("v1", "正式設定（含雨量）", dict()),
    ("v2", "預測 60 分鐘後", dict(horizon=60)),
    ("v3", "缺車門檻 10 台", dict(shortage=10)),
    ("v4", "60 分鐘後 + 門檻 10 台", dict(horizon=60, shortage=10)),
    ("v5", "不用雨量特徵", dict(features=train.BASE_FEATURES)),
]


def main():
    rows = []
    for name, desc, kw in EXPERIMENTS:
        h = kw.get("horizon", train.HORIZON)
        s = kw.get("shortage", train.SHORTAGE)
        print(f"跑 {name}：{desc} ...")
        m = train.train(h, s, official=False, features=kw.get("features"), quiet=True)
        t = m["test"]
        rows.append({
            "實驗": name, "說明": desc, "預測(分鐘)": h, "缺車門檻": s,
            "雨量特徵": "有" if any(f in m["features"] for f in train.WEATHER_FEATURES) else "無",
            "MAE_模型": t["model_mae"], "MAE_基準": t["base_mae"],
            "recall_模型": t["model_recall"], "recall_基準": t["base_recall"],
            "precision_模型": t["model_precision"], "precision_基準": t["base_precision"],
            "贏過基準": "是" if (t["model_recall"] > t["base_recall"] and t["model_mae"] <= t["base_mae"]) else "否",
            "最重要特徵": "、".join(m["top_features"][:3]),
        })
        print(f"   MAE {t['model_mae']}（基準 {t['base_mae']}），recall {t['model_recall']}（基準 {t['base_recall']}）")

    here = train.HERE
    with open(os.path.join(here, "experiments.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = train.train(train.HORIZON, train.SHORTAGE, official=False, quiet=True)
    lines = [f"\n## 實驗結果（{datetime.now():%Y-%m-%d %H:%M}）\n",
             f"資料期間 {meta['data_from']}–{meta['data_to']}，訓練 {meta['rows_train']} 筆、測試 {meta['rows_test']} 筆，"
             f"有雨量的資料 {meta['rows_with_rain']} 筆。依時間切分：前 80% 訓練、後 20% 測試。\n",
             "| 實驗 | 說明 | MAE 模型／基準 | 缺車 recall 模型／基準 | 缺車 precision 模型／基準 | 贏過基準 | 最重要特徵 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['實驗']} | {r['說明']} | {r['MAE_模型']}／{r['MAE_基準']} | {r['recall_模型']}／{r['recall_基準']} "
                     f"| {r['precision_模型']}／{r['precision_基準']} | {r['贏過基準']} | {r['最重要特徵']} |")
    lines += ["", "**選擇與理由**：（自己填：選哪一組當正式設定、為什麼）", ""]
    path = os.path.join(here, "experiments.md")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("# 實驗紀錄\n\n每次執行 `python experiments.py` 會在下面新增一節結果。基準 = 假設預測時間後車數不變（等於只看即時車數）。\n")
        f.write("\n".join(lines))
    print(f"\n已寫入 experiments.md、experiments.csv")


if __name__ == "__main__":
    main()
