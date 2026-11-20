"""
問卷分析（Forms「中大 Youbike 騎乘體驗」）

用法：
  python survey.py 問卷.xlsx                 # 印出統計數字，圖存在目前資料夾
  python survey.py 問卷.xlsx --out 圖片資料夾  # 圖存到指定資料夾

輸入：Forms →「回應」→「在 Excel 中開啟」下載的 .xlsx
注意：問卷檔含填答者的回答，不要上傳 GitHub，只提交這支程式。

輸出：
  survey_次數分布.png   借不到／還不了車的次數分布（騎乘者）
  survey_時段.png       最常發生時段（騎乘者）
  survey_沒騎原因.png   沒騎 YouBike 的原因（非騎乘者）
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 中文字型：和 analyze.py 相同
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Windows 終端機印中文不亂碼
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 用題目關鍵字找欄位（不寫死欄位順序）。每題可給多個關鍵字，題目要全部包含才算。
QUESTIONS = {
    "身分": ["身分"],                 # 第 1 題
    "天數": ["幾天"],                 # 第 2 題
    "站點": ["哪些站"],               # 第 3 題
    "沒車": ["借車", "沒車"],         # 第 4 題（第 9 題也有「沒車」，所以多加「借車」）
    "沒位": ["沒有空位"],             # 第 5 題
    "時段": ["時段"],                 # 第 6 題
    "不方便": ["不方便"],             # 第 7 題
    "原因": ["原因"],                 # 第 8 題
    "會使用": ["會使用"],             # 第 9 題
}
REQUIRED = ["身分", "天數", "沒車", "沒位", "時段", "原因", "會使用"]

# 選項（照問卷的順序排，畫圖時用；實際出現但不在清單裡的選項會自動排在後面）
COUNT_OPTIONS = ["0 次", "1–2 次", "3–5 次", "6 次以上"]
TIME_OPTIONS = ["早上 8 點前", "中午", "下午下課", "晚上", "沒遇過"]
REASON_OPTIONS = ["有其他交通工具（機車、汽車、公車）", "距離太近，走路就好",
                  "常借不到車或還不了車", "不知道站在哪裡", "天氣因素", "其他"]
IDENTITIES = ["中大學生", "中大教職員", "校外人士"]

# Forms 本身的欄位，找題目時略過
META_COLS = {"ID", "開始時間", "完成時間", "電子郵件", "名稱", "上次修改時間",
             "Id", "Start time", "Completion time", "Email", "Name", "Last modified time"}


# ---------- 小工具 ----------
def norm(s):
    """統一破折號、空白，讓「1-2 次」和「1–2 次」都對得上。"""
    s = str(s).strip()
    for dash in ["-", "－", "—", "~", "～"]:
        s = s.replace(dash, "–")
    return " ".join(s.split())


def is_blank(v):
    return pd.isna(v) or str(v).strip() == ""


def is_zero(v, unit):
    """「0 天」「0 次」（容忍「0天」「0次」）"""
    return norm(v).replace(" ", "") == f"0{unit}"


def split_multi(v):
    """Forms 複選題匯出是用分號連在一起，例如「中午;晚上;」"""
    if is_blank(v):
        return []
    return [norm(x) for x in str(v).replace("；", ";").split(";") if str(x).strip()]


def pct(n, d):
    """45%（20 人中 9 人）"""
    if d == 0:
        return "—（0 人）"
    return f"{n / d:.0%}（{d} 人中 {n} 人）"


def find_columns(df):
    cols = [c for c in df.columns if str(c).strip() not in META_COLS]
    found = {}
    for key, words in QUESTIONS.items():
        hits = [c for c in cols if all(w in str(c) for w in words)]
        if len(hits) > 1:
            print(f"⚠ 關鍵字「{'、'.join(words)}」對到多個欄位，使用第一個：{hits}")
        if hits:
            found[key] = hits[0]
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        print("找不到這些題目的欄位：", "、".join(missing))
        print("目前的欄位：", list(df.columns))
        raise SystemExit("請確認題目文字，或修改 survey.py 開頭的 QUESTIONS 關鍵字")
    return found


def count_options(series, order=None):
    """複選題：每個選項被勾幾次（依 order 排序）"""
    counts = {}
    for v in series:
        for opt in split_multi(v):
            counts[opt] = counts.get(opt, 0) + 1
    order = [norm(o) for o in (order or [])]
    # 沒人勾的選項也列出來（次數 0），圖才完整；問卷外的選項排在後面
    keys = order + sorted(k for k in counts if k not in order)
    return {k: counts.get(k, 0) for k in keys}


# ---------- 計算 ----------
def load(path):
    df = pd.read_excel(path)
    c = find_columns(df)
    df = df[~df[c["身分"]].map(is_blank).astype(bool)].copy()   # 略過整列空白
    df["_身分"] = df[c["身分"]].map(norm)
    df["_騎乘者"] = ~df[c["天數"]].map(lambda v: is_zero(v, "天")).astype(bool)
    return df, c


def stats(df, c):
    """回傳一組統計數字（全體或某身分都用這個）"""
    riders = df[df["_騎乘者"]]
    non = df[~df["_騎乘者"]]
    s = {"n": len(df), "riders": len(riders), "non": len(non)}

    # 騎乘者：第 4、5 題（略過空值）
    for key, name in [("沒車", "借不到"), ("沒位", "還不了")]:
        ans = riders[c[key]][~riders[c[key]].map(is_blank).astype(bool)]
        s[name + "_n"] = int((~ans.map(lambda v: is_zero(v, "次")).astype(bool)).sum())
        s[name + "_d"] = len(ans)

    # 騎乘者：第 6 題時段
    times = riders[c["時段"]][~riders[c["時段"]].map(is_blank).astype(bool)]
    s["時段"] = count_options(times, TIME_OPTIONS)
    s["時段_d"] = len(times)

    # 非騎乘者：第 8 題原因
    reasons = non[c["原因"]][~non[c["原因"]].map(is_blank).astype(bool)]
    s["原因"] = count_options(reasons, REASON_OPTIONS)
    s["原因_d"] = len(reasons)

    # 潛在新騎乘者：非騎乘者勾「常借不到車或還不了車」且第 9 題 ≥ 4 星
    star = pd.to_numeric(df[c["會使用"]], errors="coerce")
    shortage = non[c["原因"]].map(lambda v: any("借不到" in o or "還不了" in o for o in split_multi(v))).astype(bool)
    s["潛在新騎乘者"] = int((shortage & (star.loc[non.index] >= 4)).sum())

    # 「不知道站在哪裡」：分母 = 回答第 8 題的非騎乘者
    s["不知道站_n"] = int(non[c["原因"]].map(lambda v: any("不知道站" in o for o in split_multi(v))).astype(bool).sum())

    # 第 9 題平均星數
    s["星_全體"] = star.mean(), int(star.notna().sum())
    s["星_騎乘者"] = star.loc[riders.index].mean(), int(star.loc[riders.index].notna().sum())
    s["星_非騎乘者"] = star.loc[non.index].mean(), int(star.loc[non.index].notna().sum())
    return s


def fmt_star(t):
    mean, n = t
    return f"{mean:.2f} 星（{n} 人）" if n else "—（0 人）"


def print_stats(s, title):
    print(f"\n===== {title} =====")
    print(f"回收份數：{s['n']}　騎乘者：{s['riders']}　非騎乘者：{s['non']}")
    print("【騎乘者】")
    print(f"  過去兩週曾借不到車：{pct(s['借不到_n'], s['借不到_d'])}")
    print(f"  過去兩週曾還不了車：{pct(s['還不了_n'], s['還不了_d'])}")
    print(f"  發生時段（複選，{s['時段_d']} 人回答）：")
    for k, v in s["時段"].items():
        print(f"    {k}：{v} 次（{pct(v, s['時段_d'])}）")
    print("【非騎乘者】")
    print(f"  沒騎的原因（複選，{s['原因_d']} 人回答）：")
    for k, v in s["原因"].items():
        print(f"    {k}：{pct(v, s['原因_d'])}")
    print(f"  勾「不知道站在哪裡」：{pct(s['不知道站_n'], s['原因_d'])}")
    print(f"  潛在新騎乘者（勾缺車且第 9 題 ≥ 4 星）：{s['潛在新騎乘者']} 人")
    print("【第 9 題：會不會使用預測（1–5 星）】")
    print(f"  全體：{fmt_star(s['星_全體'])}　騎乘者：{fmt_star(s['星_騎乘者'])}　非騎乘者：{fmt_star(s['星_非騎乘者'])}")


def print_open(df, c):
    if "不方便" not in c:
        return
    print("\n===== 第 7 題開放回答（給簡報挑故事用） =====")
    skip = {"沒有", "無", "沒", "沒遇過", "none", "no"}
    answers = [str(v).strip() for v in df[c["不方便"]]
               if not is_blank(v) and str(v).strip().strip("。.").lower() not in skip]
    if not answers:
        print("（沒有）")
    for i, a in enumerate(answers, 1):
        print(f"{i}. {a}")


# ---------- 畫圖 ----------
def bar_labels(ax, bars, denom):
    for b in bars:
        h = b.get_height()
        label = f"{int(h)}" if denom == 0 else f"{int(h)}（{h / denom:.0%}）"
        ax.annotate(label, (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9)


def plot_counts(df, c, out):
    riders = df[df["_騎乘者"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    for i, (key, name) in enumerate([("沒車", "想借車但沒車"), ("沒位", "想還車但沒空位")]):
        ans = riders[c[key]][~riders[c[key]].map(is_blank).astype(bool)].map(norm)
        order = [norm(o) for o in COUNT_OPTIONS]
        order += sorted(set(ans) - set(order))
        vals = [int((ans == o).sum()) for o in order]
        xs = [x + (i - 0.5) * width for x in range(len(order))]
        bars = ax.bar(xs, vals, width, label=f"{name}（N = {len(ans)}）")
        bar_labels(ax, bars, len(ans))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("人數")
    ax.set_title(f"過去兩週借不到／還不了車的次數（騎乘者 N = {len(riders)}）")
    ax.legend()
    save(fig, out, "survey_次數分布.png")


def plot_options(counts, denom, title, xlabel, out, name, horizontal=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys, vals = list(counts.keys()), list(counts.values())
    if horizontal:   # 原因選項字比較長，用橫條
        bars = ax.barh(keys[::-1], vals[::-1])
        for b in bars:
            w = b.get_width()
            label = f"{int(w)}" if denom == 0 else f"{int(w)}（{w / denom:.0%}）"
            ax.annotate(label, (w, b.get_y() + b.get_height() / 2), va="center", fontsize=9,
                        xytext=(3, 0), textcoords="offset points")
        ax.set_xlabel("勾選人數（複選）")
    else:
        bars = ax.bar(keys, vals)
        bar_labels(ax, bars, denom)
        ax.set_ylabel("勾選人數（複選）")
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    save(fig, out, name)


def save(fig, out, name):
    for ax in fig.axes:   # 人數只顯示整數刻度
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) if ax.get_ylabel() else ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"圖已存：{path}")


# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser(description="Forms 問卷分析")
    ap.add_argument("xlsx", help="Forms 匯出的 Excel 檔")
    ap.add_argument("--out", default=".", help="圖片存放資料夾（預設：目前資料夾）")
    args = ap.parse_args()

    df, c = load(args.xlsx)
    os.makedirs(args.out, exist_ok=True)

    # 各身分人數
    print(f"回收份數：{len(df)}")
    for ident in IDENTITIES + sorted(set(df["_身分"]) - set(IDENTITIES)):
        print(f"  {ident}：{int((df['_身分'] == ident).sum())} 人")

    s = stats(df, c)
    print_stats(s, "全體")
    for ident in IDENTITIES:
        sub = df[df["_身分"] == ident]
        if len(sub):
            print_stats(stats(sub, c), ident)
    print_open(df, c)

    print()
    plot_counts(df, c, args.out)
    plot_options(s["時段"], s["時段_d"], f"借不到／還不了車最常發生的時段（騎乘者 N = {s['時段_d']}）",
                 "時段", args.out, "survey_時段.png")
    plot_options(s["原因"], s["原因_d"], f"沒有（或很少）騎 YouBike 的原因（非騎乘者 N = {s['原因_d']}）",
                 "", args.out, "survey_沒騎原因.png", horizontal=True)


if __name__ == "__main__":
    main()
