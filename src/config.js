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
    corridorSoftness: 0.8,      // 走廊外的高斯尾巴（以半寬為單位）
    minCorridorWeight: 0.12,    // 低於此權重才真正排除
    horizonFallback: 0.45,      // 無 IMU 重力向量時的地平線 y 比例
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
    resampleMs: 600,
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
    // 佐證：影像上前車應同時往地平線方向移動
    requireUpwardMotion: true,
    upwardAgreeRatio: 0.55,
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
