# 中大 YouBike 缺車／滿柱預警

> 用機器學習預測中央大學 YouBike 站點 **30 分鐘後** 的可借車數，預計缺車時主動通知學生，並建議提早出門、改去附近有車的站或改搭公車。

Microsoft × NCU AI 創新應用主題挑戰賽（AI for Better Campus & Life）參賽作品｜國立中央大學機器學習課程專題

---

## 為什麼做這個

中大學生在上下課尖峰時段，常常走到 YouBike 站才發現**沒車可借或沒位可還**，導致通勤時間增加，甚至上課遲到。

現有的查詢方式（包括官方 App）主要顯示**當下**的車數。但學生出門時真正需要知道的是：**等我走到站的時候，還有沒有車？**

## 我們怎麼做

| 一般查詢 | 本系統 |
|---|---|
| 自己打開 App，看到「現在」幾台 | 預計缺車時**主動通知** |
| 只告訴你這站有幾台 | 建議**提早出門、改去附近的站或改搭公車** |
| 一般時段資料 | 模型特徵放進**中大的上課節次**，專門針對校園通勤尖峰 |

## 系統架構

```mermaid
flowchart LR
    A[桃園市 YouBike 2.0<br/>即時開放資料] -->|每 10 分鐘| B[collector.py<br/>收資料]
    B --> C[(data/<br/>每日 CSV)]
    C --> D[train.py --auto<br/>LightGBM 自動訓練]
    D --> E[(model.txt)]
    C --> F[notify.py<br/>預測 30 分鐘後車數]
    E --> F
    F --> G[(predictions.xlsx)]
    F -->|中大站點預測缺車| H[Power Automate<br/>webhook]
    H --> I[Email 缺車預警]
    G --> J[Copilot Studio<br/>通勤小幫手]
```

全部流程在 **GitHub Actions** 上自動執行：每次連續工作約 5.5 小時、每 10 分鐘一輪，結束前自動啟動下一輪，不需要任何人的電腦開著。

| 環節 | 檔案 | 執行方式 |
|---|---|---|
| 收資料 | `collector.py` | GitHub 自動，每 10 分鐘 |
| 訓練正式模型 | `train.py --auto` | GitHub 自動：資料滿 7 天後開始，約每週一次；改了正式設定也會重訓 |
| 預測與通知 | `notify.py` | GitHub 自動，每 10 分鐘 |
| 缺車通知 | Power Automate（收到 Teams webhook 要求時 → 傳送電子郵件） | 自動 |
| 對話小幫手 | Copilot Studio，讀取 `predictions.xlsx` | Demo 前手動更新資料 |
| 本機實驗 | `train.py --horizon … --shortage …` | 手動，結果存成 `model_local.txt`，不上傳 |
| 簡報素材 | `analyze.py`、`nearby.py` | 手動 |

## 模型

- **預測目標**：各站 30 分鐘後的可借車數；可借車數 ≤ 2 台視為「缺車」。
- **模型**：LightGBM 迴歸。
- **特徵**：站點、時段、星期、是否週末、**距離下一節課開始的分鐘數**（中大上課節次）、目前車數與總車柱、過去 10 / 20 / 30 / 60 分鐘的車數與變化。
- **基準（baseline）**：假設 30 分鐘後車數不變，也就是只看當下的即時車數。
- **評估**：依時間切分，前 80% 訓練、後 20% 測試，避免模型偷看到未來。指標為 MAE，以及缺車預警的 precision、recall。
- **保護機制**：新模型的缺車 recall 要高於基準、MAE 不能比基準差，才會取代正式模型。

### 結果

> 待 11/23 後以真實資料訓練並完成實驗後更新（實驗紀錄見 `experiments.md`）。

| | MAE（台） | 缺車 precision | 缺車 recall |
|---|---|---|---|
| 基準（不變） | | | |
| LightGBM | | | |

## 通知的保護機制

- 同一站一小時內只通知一次
- 06:30 前、22:00 後不通知
- 最新資料超過 30 分鐘（例如收集中斷）就不通知，避免用舊資料誤報
- 只通知中大校內站點（`collect.yml` 的 `WATCH_KEYWORDS` 設定）

## 與其他做法的比較

**台大學生開發的「Ubike」App**：除了顯示各站車數，還能預測熱門站點的等車時間，做法是依據前一小時的租借情況推估（[INSIDE 報導](https://www.inside.com.tw/article/2833-ubike-app-case-study)）。

本作品的不同之處：

| | 台大學生的 Ubike App | 本作品 |
|---|---|---|
| 預測方式 | 用**前一小時**的租借情況推估 | 用持續累積的歷史資料訓練**機器學習模型**，並與「不變」基準比較、量化準確度 |
| 預測內容 | 熱門站點的等車時間 | 30 分鐘後的可借車數與缺車預警 |
| 校園情境 | 一般使用者 | 加入**中大上課節次**特徵，針對早八與下課尖峰 |
| 使用方式 | 使用者主動打開 App 查詢 | 預計缺車時**主動通知**，並由 Copilot 小幫手建議替代方案 |
| 運作方式 | App | 雲端全自動：收資料、重訓、預測、通知都不需人工 |

「預測車況」本身不是新點子，本作品的重點是**把預測轉成可以行動的決策建議，並主動送到學生手上**。

此外，北北桃曾規劃於 2026 年 1 月 1 日至 6 月 30 日試辦 YouBike「友愛接力」計畫：在無位可還的站借車，或在無車可借的站還車，可以獲得騎乘券；營運單位也表示即使增加調度，尖峰時段仍偶有無車可借或無柱可還的情況（[中央社報導](https://www.cna.com.tw/news/ahel/202512110100.aspx)）。本系統的預測可以提前告訴使用者哪裡快沒車、哪裡快滿柱，與這類調度政策互相配合。

## 擴充性

系統目前收集**整個中壢區**的站點資料，模型也是用中壢區所有站訓練。擴充只需要改設定，不需要重寫程式：

| 階段 | 範圍 | 要改什麼 |
|---|---|---|
| 現在 | 中大校內 2 站 | — |
| 下一步 | 中大通勤圈（周邊 1–1.5 公里） | `WATCH_KEYWORDS` 加入周邊站名 |
| 中期 | 中壢其他校園，例如中原大學 | 加關鍵字，`CLASS_STARTS` 換成該校節次 |
| 長期 | 整個桃園市，提供營運單位調度參考 | `collector.py` 拿掉 `--area 中壢區` |

## 在自己電腦上執行

```bash
pip install -r requirements.txt

python train.py                              # 用正式設定訓練（存成 model_local.txt，不上傳）
python train.py --horizon 60 --shortage 3    # 實驗：改預測時間與缺車門檻
python train.py --predict                    # 用正式模型預測，輸出 predictions.xlsx
python analyze.py shortage --keyword 中央大學 # 各時段缺車時間比例
python analyze.py plot --keyword 依仁堂 --official   # 實際 vs 預測對照圖
python nearby.py --km 1.5                    # 中大周邊站點清單
```

正式設定（`HORIZON`、`SHORTAGE`、`CLASS_STARTS`）寫在 `train.py` 最上方。修改後提交並同步，GitHub 會自動用新設定重訓；也可以到 Actions 按「立即重新訓練」。

## GitHub Actions

| Workflow | 用途 |
|---|---|
| `collect.yml` | 主流程：收資料 → 自動訓練 → 預測 → 通知，接力不中斷 |
| `train-now.yml` | 手動立即重新訓練 |
| `test-notify.yml` | 送一則測試通知，確認通知流程正常 |

需要在 **Settings → Secrets and variables → Actions** 設定 `WEBHOOK_URL`（Power Automate 流程的網址）。

## 資料來源與授權

- 桃園市政府交通局「桃園公共自行車即時服務資料」（[政府資料開放平臺](https://data.gov.tw/dataset/28228)），依《政府資料開放授權條款－第 1 版》使用。
- 只使用站點層級的車數資料，不含任何個人資料。

## 限制

- 模型只能預測約 30 分鐘後，無法預測隔天或更遠的時間。
- Copilot Studio 小幫手的資料需要手動更新。
- 預測可能有誤差，通知中會提醒使用者以現場為準。

## 團隊

| 成員 | 負責 |
|---|---|
| | |
| | |
| | |
| | |
