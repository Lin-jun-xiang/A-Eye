// =============================================
// A-Eye v7 — 全部可調參數集中於此
// =============================================
// 設計原則：這裡只允許兩種數字
//   (a) 物理量（公尺、秒、m/s、弧度）—— 有真實世界意義
//   (b) 機率（誤警率 α、漏報率 β）—— 有統計意義
// 「像素」「百分比」「連續幾幀」這類拍腦袋的數字一律避免；
// 需要尺度時，一律用資料自己估出來的雜訊尺度 σ 做正規化。

export const CONFIG = {
  // ---------- 迴圈與節流 ----------
  loop: {
    detectHz: 8,        // YOLO 提交上限（實際受推論速度限制，會自動掉幀不排隊）
    lightHz: 5,         // 紅綠燈色彩分析
    flowHz: 25,         // 光流：跟著 video frame 跑，這是上限
  },

  // ---------- 相機 ----------
  camera: {
    width: 1280,
    height: 720,
    // 光流的 ROI 最長邊上限（px）。ROI 以「原生解析度」裁切後最多縮到這個大小，
    // 絕不整幀降採樣 —— 訊號強度與解析度成正比。
    roiMaxSide: 384,
  },

  // ---------- 物件偵測 ----------
  yolo: {
    // 依序嘗試；384 版最快，沒有就退回既有檔案
    modelCandidates: [
      './models/yolov8n_384.onnx',
      './models/yolov8n.onnx',
      './models/yolov8s.onnx',
    ],
    // 模型輸入為動態軸時採用；靜態軸則以模型自身尺寸為準
    preferredInputSize: 384,
    confThreshold: 0.30,
    iouThreshold: 0.45,
    // 只解碼我們要的類別 → 解碼迴圈從 8400×80 降到 8400×4
    keepClasses: [2, 5, 7, 9],   // car, bus, truck, traffic light
    providers: ['webgpu', 'wasm'],

    // onnxruntime-web 的載入來源。
    // 主執行緒會先用 fetch() 抓下來轉成同源 blob URL 再交給 worker，
    // 因為 worker 直接 importScripts() 跨來源網址在 WebKit 上會失敗
    // （若有 Service Worker 攔截，回應會變成 opaque，而規範禁止
    //   importScripts 接受 opaque response）。抓不到才退回直接載入。
    ortVersion: '1.20.1',
    ortFiles: [
      'ort.all.min.js',     // 含 webgpu + wasm
      'ort.min.js',         // 預設 bundle
      'ort.wasm.min.js',    // 純 wasm
    ],
  },

  vehicleClasses: [2, 5, 7],
  lightClass: 9,

  // ---------- 車道線 ----------
  // v7 已移除。v6 設定的 models/ufld_tusimple.onnx 其實從未存在，
  // useUfld 一直是 false —— 車道線功能從來沒有真正運作過。
  // 而 TuSimple 的訓練視角與手機任意安裝角度差異太大，硬接回來只會誤刪真前車。
  // 現在由「IMU 重力向量推得的地平線 + 透視走廊」取代（見 frontCar 區塊）。
  // 若日後要加回：FrontCarSelector.select() 已預留 laneWeightFn 參數，
  // 且必須只做軟性加權，不得參與硬性排除。

  // ---------- 多目標追蹤 ----------
  tracker: {
    iouGate: 0.10,          // 關聯門檻（配合 KF 預測框，可以放很寬）
    // 允許用「中心距離 / 尺寸一致性」補救 IoU 掉到 0 的情況
    centerGateRatio: 1.2,   // 中心距離 ≤ 1.2 × 框對角線 → 仍可關聯
    scaleGate: 2.2,         // 面積比在 1/2.2 ~ 2.2 之間才可關聯
    confirmHits: 2,         // 連續命中數 → 確認為正式 track
    maxCoastMs: 700,        // 目標暫時消失後仍以 KF 慣性維持的時間
    // KF 過程雜訊（物理意義：影像座標的加速度強度）
    qCenter: 4e4,           // px²/s³
    qLogSize: 0.8,          // (log px)²/s³
    rCenter: 25,            // 偵測框中心量測雜訊 px²
    rLogSize: 2e-3,         // 偵測框 log 尺寸量測雜訊
  },

  // ---------- 前車選取 ----------
  frontCar: {
    minAreaRatio: 0.004,        // 面積佔比下限（太小的遠車不可靠）
    // 「最前方」判據 = bbox 底邊最低（透視幾何），不再用 MiDaS
    //
    // 走廊寬度由地面平面幾何嚴格推導，不是憑感覺的畫面比例：
    //   影像中 y 處的地面距離   Z = f·h / (y − y_horizon)
    //   橫向 X 公尺投影成像素   X_px = f·X / Z = X·(y − y_horizon) / h
    //   → 焦距 f 自然消掉！走廊半寬（像素）＝ (laneHalfWidthM / cameraHeightM) · (y − y_horizon)
    // 於是只需要兩個真實物理量，而且在地平線處自動收斂到 0。
    laneHalfWidthM: 1.9,        // 車道半寬 + 邊際（公尺）
    cameraHeightM: 1.2,         // 手機安裝高度（公尺）
    vehicleWidthM: 1.8,         // 一般車輛車寬（公尺），用於幾何自洽檢定
    corridorSoftness: 0.8,      // 走廊外的高斯尾巴（以半寬為單位）
    minCorridorWeight: 0.12,    // 低於此權重才真正排除
    horizonFallback: 0.45,      // 無 IMU 重力向量時的地平線 y 比例

    // ---- 幾何可信度（軟性加權，不是硬性排除）----
    // 自車結構（引擎蓋 / 儀表板 / 反光）會被 YOLO 認成 car，而它的底邊在畫面
    // 最下方 → proximity 分數必定最高 → 永遠被選成前車。
    //
    // 但「畫面最下緣就排除」是錯的：手機架高看不到引擎蓋時，那條規則會把
    // 3 公尺內、輪胎接地點被畫面裁掉的真前車也排除掉。
    //
    // 正確的依據是「這個框像不像一台站在地面上的車」，而且要用實體尺寸推導：
    //
    //   由底邊：Z = f·h_cam / (y_bottom − y_horizon)
    //   由寬度：Z = f·W_car  / w_px
    //   兩者相除 → w_px / Δy = W_car / h_cam     ← 焦距 f 自己消掉了！
    //
    // 所以「寬度 ÷ 底邊到地平線的距離」對地面上的任何車輛都是同一個常數，
    // 與距離無關、與焦距無關。自車結構不站在地面上 → 這個比值會偏高
    //（對它的底邊位置而言「太寬了」）。
    //
    // 而且連 W_car/h_cam 都不寫死：取畫面上所有車輛 track 的比值中位數當中心，
    // 用「離群幾倍」判定。這樣 cameraHeightM 填錯也不會壞。
    plausibility: {
      // 車尾長寬比先驗（實體尺寸推導：車尾寬 1.4~2.6m、高 1.2~3.2m
      // → 0.44~2.17；再放寬到 3.2 容納斜看時 bbox 含車側的情形）
      aspectMin: 0.45,
      aspectMax: 3.2,
      aspectSoftness: 0.35,     // 超出區間後的高斯尾巴（單位就是長寬比本身）
      // w/Δy 只檢查「高側」離群：自車結構偏高，而機車偏低 ——
      // 只砍高側就不會誤殺機車（單側檢定是刻意的）
      // 1.3：大車（W=2.5m）約是族群中位數的 1.39 倍、斜看時更高，
      // 所以這裡刻意只扣一點分；真正把自車結構壓下去的是「三項相乘」
      //（長寬比 × w/Δy × 沒有剎車燈），沒有任何單一項是決定性的。
      ratioOutlierFactor: 1.3,
      ratioSoftness: 0.25,
      // 參考值直接用 vehicleWidthM / cameraHeightM（兩個真實可量的公尺數）。
      // 族群中位數目前只用來「觀察」——印在除錯面板上，讓實車影片告訴我們
      // 真實族群是否與設定值一致；等有資料再決定要不要改成自動校準。
      // 不現在就自動校準的理由：紅燈前方常常只有「前車 + 引擎蓋」兩個框，
      // 兩者各佔一半樣本時中位數會落在兩者之間，離群值反而變成中心。
      ratioMinSamples: 12,      // 中位數要幾個樣本才顯示（不足時面板顯示 --）
      ratioSampleGapMs: 800,    // 同一個 track 最快多久貢獻一次樣本
      ratioMaxSamples: 128,
      // 已觀察一段時間卻始終沒看到剎車燈 → 降權（不是排除）。
      // 紅燈停等時前車的剎車燈幾乎必然亮著，所以「看得到剎車燈」是
      // 比任何幾何規則都直接的正向證據，而且與手機安裝方式完全無關。
      noLampWeight: 0.35,
      noLampAfterMs: 1500,
    },

    // 自車結構（引擎蓋 / 儀表板 / A 柱反光）會被 YOLO 誤判成 car，
    // 而且它的 bbox 底邊在畫面最下方 → proximity 分數必定最高 → 永遠被選成前車，
    // 於是光流量的是一個「永遠不動的東西」，起步警示結構上不可能觸發。
    //
    // 判別依據是物理性質，不是畫面比例：自車結構相對相機完全靜止，
    // 所以「自車行駛中，框卻完全不動」只可能是自車的一部分
    // （真正停著的路邊車在自車行駛時 bbox 會持續放大）。
    // 只在自車行駛時學習 —— 紅燈停車時前車本來就不動，否則會把真前車列入黑名單。
    egoStructure: {
      learnWhileMovingMs: 2000,   // 行駛中連續「完全不動」超過此時間 → 判為自車結構
      frozenLogScaleRate: 0.06,   // |d log(尺寸)/dt| 上限（1/s）
      frozenVyPxPerS: 8,          // |中心垂直速度| 上限（px/s）
      matchIou: 0.55,             // 與已學到的區域 IoU 超過此值 → 直接排除
      maxRegions: 6,
    },
  },

  // ---------- 光流 ----------
  flow: {
    // OpenCV.js 來源（~10MB）。docs.opencv.org 沒有送 CORS 標頭，
    // 所以只能用 <script> 標籤載入（no-cors），不能用 fetch。
    // 依序嘗試，避免單一網址失效就整個起步偵測停擺。
    cvUrls: [
      'https://docs.opencv.org/4.9.0/opencv.js',
      'https://docs.opencv.org/4.10.0/opencv.js',
      'https://docs.opencv.org/4.8.0/opencv.js',
    ],
    maxFgPoints: 80,
    maxBgPoints: 100,
    winSize: 21,
    pyrLevels: 3,
    minFgPoints: 12,
    minBgPoints: 10,
    resampleMs: 900,
    // 量測基線（毫秒）。訊號 ∝ dt，雜訊（LK 的次像素誤差）與 dt 無關，
    // 所以「每幀比一次」是 SNR 最差的做法：40ms 內尺度只變 0.1%。
    // 改成「每 baselineMs 產生一筆量測，且相鄰量測不重疊」——
    // SNR 提升約 6 倍，同時保住 SPRT 需要的觀測獨立性
    // （重疊基線會讓相鄰觀測高度相關，證據被重複計算，那正是 v6 的老毛病）。
    baselineMs: 240,
    roiInflate: 1.7,        // ROI = bbox × 1.7（外圈當背景環）
    fgShrink: 0.10,         // 前景取 bbox 內縮 10%（避開邊緣混入背景）
    fbErrorPx: 1.0,         // forward-backward 一致性上限（低解析度像素）
    ransacReprojPx: 2.0,
    minBgInlierRatio: 0.5,  // 背景 inlier 比例過低 → 該 tick 不可信
  },

  // ---------- 起步判定 ----------
  departure: {
    // 觀測量：V = −d·log(尺度)/dt，單位 1/秒，物理上就是 1/TTC
    // KF 過程雜訊：V 的隨機遊走強度 (1/s)²/s
    qV: 0.02,
    // 判定門檻（機率，不是像素）
    alpha: 1e-4,            // 每次判定可容忍的誤警率 → z_fire = 3.72σ
    beta: 0.05,             // 漏報率（用於 SPRT 門檻）
    effectSize: 2.0,        // H1 的效應量（以每 tick 標準差為單位）
    zClamp: 3.0,            // 單 tick 的 z 上限 —— 防止單一離群 tick 獨力觸發
    // 物理最小值：1/TTC 低於此值視為「幾乎沒動」，不算起步
    // 0.02 /s ⇔ TTC 50 秒 ⇔ 10 公尺外以 0.2 m/s 遠離
    minInvTtc: 0.02,
    dwellMs: 260,           // 證據需持續多久（防模型誤差造成的瞬時尖峰）
    minTicks: 4,
    // 已有獨立證據（剎車燈熄滅）時放寬 —— dwell 與 minTicks 的存在理由是
    // 「防止模型誤差造成的瞬時尖峰獨力觸發」，而一個獨立來源的佐證
    // 正當地降低了所需的自我佐證量。這不是偷跑，是貝氏更新。
    dwellMsPrimed: 120,
    minTicksPrimed: 2,
    // 目標必須已被連續追蹤這麼久才「武裝」。
    // 前車起步的前提是前車先在那裡；鑽車縫掠過的機車、切進來的車
    // 都是短暫出現就持續遠離，不加這道前提就會變成誤報來源。
    armMs: 500,
    // 佐證：影像上前車應同時往地平線方向移動
    requireUpwardMotion: true,
    upwardAgreeRatio: 0.55,
    // 只有位移量大於此值的量測才拿來投票 —— 次像素雜訊的符號等於擲硬幣，
    // 讓它進投票池只會把比例永遠壓在 0.5 附近，變成一道隨機閘門。
    upwardMinPx: 0.4,
    upwardWindow: 10,
    // 兩個獨立判據（KF z 檢定 與 tick 級 SPRT）都成立才觸發
    requireBoth: true,
    llrDecayTau: 1.2,       // SPRT 證據的時間常數（秒）
    coastMs: 700,           // 目標短暫遺失時保留證據的時間（不再歸零！）
    cooldownMs: 5000,
  },

  // ---------- 自車運動 ----------
  ego: {
    gpsMoveSpeed: 1.5,      // m/s，遲滯上緣
    gpsStillSpeed: 0.5,     // m/s，遲滯下緣
    gpsStaleMs: 4000,
    // IMU：以「靜止時學到的加速度變異數基線」為尺度，超過倍數 → 移動
    imuMoveFactor: 6.0,
    imuWindowMs: 1200,
    // 視覺：背景點的中位位移（已扣陀螺儀旋轉）大於雜訊尺度數倍 → 移動
    visualMoveSigma: 4.0,
    unknownIsStill: false,  // 三路全不可用時，寧可靜默也不要亂報
  },

  // ---------- IMU ----------
  imu: {
    bufferMs: 2000,
    // 陀螺儀 → 畫面位移的增益由 RLS 線上學習，這是可用門檻
    calibMinQuality: 0.55,
    calibMinSamples: 60,
    // 若學到的增益與視覺觀測嚴重不符 → 該 tick 的背景估計不可信
    disagreeSigma: 4.0,
  },

  // ---------- 剎車燈（快路徑）----------
  // 為什麼值得做：任何基於運動的量測都必須「等車真的動了」才有訊號，
  // 但駕駛鬆開剎車踏板到車輛實際移動之間有 0.3~1 秒 —— 剎車燈在鬆踏板
  // 的那一瞬間就熄了。這是唯一能讓警示「本質上變快」的訊號源。
  //
  // 而且它是光度訊號不是幾何訊號：5~10m 的剎車燈在影像上是數十像素的
  // 飽和紅光，而同一情境下 240ms 的尺度變化只有 2.4% —— 差一個數量級。
  //
  // 判定一律用「相對於這台車自己」的變化，不用絕對色彩門檻
  //（絕對門檻換個曝光、換個燈型就失效，這是紅綠燈那條路的教訓）。
  brakeLight: {
    hz: 10,                   // 分析頻率上限
    maxSide: 160,             // 裁切分析用的最長邊（px）

    // ROI 是「車尾燈在車尾的哪個位置」的實體先驗，不是畫面比例
    sideFrac: 0.30,           // 左右燈區各佔 bbox 寬的比例
    centerFrac: 0.30,         // 中央車身參考區佔寬的比例
    yTop: 0.38,               // 燈區的垂直範圍（bbox 高的比例）
    yBottom: 0.90,
    topKFrac: 0.12,           // 每個 ROI 只取「最紅的前 12%」像素平均
                              // （燈只佔 ROI 一部分，取全區平均會被車身稀釋）

    minContrast: 1.8,         // 燈區紅度 / 車身紅度 → 排除「紅色車身」
                              // 這一項判定的是「有一對紅燈」（尾燈也算），
                              // 不是「剎車燈亮」—— 兩者必須分開，見 onRatio
    onRatio: 0.70,            // 亮度達到自身峰值的此比例 → 視為剎車中。
                              // 夜間尾燈長亮、剎車燈是同一燈室變更亮，
                              // 靠「相對自己的峰值」才分得開（絕對亮度分不開）
    symmetryTol: 0.85,        // |ln(左/右)| 上限 → 排除方向燈（單側）
    offRatio: 0.45,           // 掉到峰值的此比例以下 → 視為熄滅
    offConfirmMs: 350,        // 熄滅需持續多久（腳在踏板上微動會短暫閃熄）
    peakHalfLifeMs: 20000,    // 峰值參考值的半衰期
    overexposedFrac: 0.35,    // 過曝像素比例超過此值 → 回報 unknown 而非 off
                              // 關鍵：失效模式必須是「不知道」，否則夕陽直射會誤報
    blinkWindowMs: 3000,      // 雙閃偵測窗口
    blinkMinCycles: 2,        // 窗口內亮熄循環數 → 判為閃爍（雙閃/方向燈），暫停判定

    // 「熄燈」給起步判定的先驗，用兩個可估計的機率表達（不是拍腦袋的分數）
    pOffGivenDepart: 0.90,    // 即將起步時，剎車燈會熄的機率
    pOffGivenStay: 0.02,      // 不起步卻熄燈（打 P/N、腳移開）的機率
    // → 先驗 LLR = ln(0.90/0.02) ≈ 3.81
    priorValidMs: 4000,       // 先驗的有效期：熄燈後多久內仍算「已預備」

    // 直接對「鬆剎車」本身發一個較輕的提示（不等運動確認）。
    // 代價是打 P/N 會誤報一次 —— 這是刻意接受的取捨：塞車與紅燈情境下
    // 打 P/N 的機率極小，而搶到的 0.3~1 秒是這個 App 的全部價值。
    alertOnRelease: true,
  },

  // ---------- 紅綠燈 ----------
  light: {
    confirmMs: 400,
    cooldownMs: 6000,
    minBrightPixels: 8,
  },

  // ---------- 警示 ----------
  alert: {
    screenOffTimeoutMs: 5 * 60 * 1000,
  },

  // ---------- 除錯 ----------
  debug: {
    logEvents: true,    // 觸發警示時把當下的判定統計量印到 console
  },
};

/** 由 α / β 推導出來的門檻（避免在程式各處重算） */
export function derivedThresholds(cfg = CONFIG) {
  const d = cfg.departure;
  return {
    zFire: -normInvLocal(d.alpha),                       // 例：α=1e-4 → 3.719
    sprtA: Math.log((1 - d.beta) / d.alpha),             // 上門檻 → 觸發
    sprtB: Math.log(d.beta / (1 - d.alpha)),             // 下門檻 → 歸零
  };
}

// 避免 config 依賴 util（保持 config 為葉節點模組）
function normInvLocal(p) {
  // 只需要小 p 的尾端，用 Acklam 的低尾分支即可
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
  const q = Math.sqrt(-2 * Math.log(p));
  return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}
