# 離線 pipeline 跑機（不需要瀏覽器）

用 node 直接跑 `src/core/pipeline.js`，把每一步的內部狀態印出來。
與 `replay.html` / `analyze.html` 跑的是**同一份 pipeline**，差別只在執行環境：
這裡用 `onnxruntime-node` 跑 YOLO、用 node 載入 `opencv.js`。

## 為什麼需要它

在這之前，每一輪除錯都得「使用者用手機路測 → 貼除錯面板的截圖」。
一次往返成本很高，而且只看得到當下那一格數字 —— 沒辦法回答
「這 18 秒裡光流失敗了幾次、各是什麼原因」這種問題。

2026-09-09 的除錯就是靠它才找到真正的根因：夜間近距離的車尾在原始灰階上
**只撒得出 1 個特徵點**（下限 12），於是 `needAnchor` 永遠成立、
240ms 基線永遠完成不了、起步判定永遠收不到量測。
在瀏覽器裡只看得到「證據 0%」，看不到這一層。

## 前置作業

兩個依賴都**不進 repo**，避免讓這個專案背上 npm 依賴：

```bash
cd tools/offline
npm init -y
npm i onnxruntime-node                                  # 約 100MB
curl -o opencv.js https://docs.opencv.org/4.9.0/opencv.js   # 約 10MB
```

`models/yolov8n.onnx` 已經在 repo 裡，不需要另外準備。

## 用法

```bash
# 分析 in.mp4 的第 148~166 秒，每秒取 10 幀
ffmpeg -v error -i in.mp4 -ss 148 -t 18 \
  -vf "fps=10,crop=588:940:0:0" -pix_fmt rgba -f rawvideo pipe:1 \
  | node run.mjs 588 940 10 148
```

四個位置參數是 `<寬> <高> <fps> <起始秒>`，必須與 `-vf` 的 crop/fps 一致
（跑機沒辦法從 rawvideo 得知尺寸）。

環境變數：

| | |
|---|---|
| `ASSUME_STILL=1` | 強制假設自車靜止。不設的話由背景尺度變化率自己判斷 |
| `CV_JS=...` | opencv.js 的路徑（預設 `./opencv.js`） |
| `MODEL=...` | onnx 模型路徑（預設 `../../models/yolov8n.onnx`） |

**注意 crop**：如果來源是「app 執行畫面的螢幕錄影」，要把 UI 面板裁掉
（例如 588x1280 的錄影，`crop=588:940:0:0` 只留影片區）。
這種輸入與 app 實際餵給 YOLO 的相機幀並不完全相同（水平被 `object-fit: cover`
裁掉一部分、又多經過一次 H.264 壓縮），所以偵測率是「不高於」實際值的估計。

## 輸出

每秒印一行內部狀態，最後是總結：

```
t=157.0s  target=#12 331x240  新鮮61%  ego=still  z=2.25/3.72 LLR=2.1/9.2 V=0.002 n=2
          below-min-ttc  flow=ok fg=48/62 bg=31/44  剎車燈=on(第三燈on)

===== 總結 =====
幀數 180（148~165.9s @10fps）
偵測 90 次，偵測框 141 個 → 配對 120 新建 21 淘汰 19
目標新鮮 62%（107/172）　換手 15 次（清空證據 15 次）
光流 ok 21　失敗：reanchor:64 accumulating:59 fg-too-few:15 bg-too-few:9
事件 0 個
```

怎麼讀：

| 欄位 | 看什麼 |
|---|---|
| `偵測框 → 配對 / 新建` | 偵測框少 → 偵測器的問題；框多但配對少 → 關聯門檻的問題 |
| `目標新鮮 %` | 目標有多少比例的 tick 是剛被偵測更新過的（其餘是 KF 外推） |
| `換手 N 次（清空證據 M）` | 前車選取有多不穩。每次清空都會讓證據從零開始 |
| `光流 ok / 失敗原因` | `fg-too-few` = 目標沒有紋理；`bg-too-few` = 背景沒有紋理；`reanchor` 過多通常是前兩者造成的連鎖反應 |
| `flow=ok fg=a/b bg=c/d` | 通過檢核的點數 / 撒出的點數 |

## 三個會靜默無限等待的坑

如果自己改這個跑機，這三個都會讓程式卡住而不報錯：

1. **`await cv`** —— opencv.js 的 Module 有一個相容性用的 `then()`，
   它用 Module 自己來 resolve，而 Module 又是 thenable → Promise 機制無限遞迴。
2. **async function 裡 `return cv`** —— 同一個坑的第二個化身：
   async 的 promise 會用回傳值 resolve，於是又去呼叫 `cv.then()`。
3. **只掛 `onRuntimeInitialized`** —— runtime 可能在掛上 handler 之前就初始化完成，
   那時 callback 永遠不會被呼叫。必須同時輪詢 `cv.Mat`。
