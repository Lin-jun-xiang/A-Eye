// ===== A-Eye v6 — YOLOv8n + MiDaS + GPS + 靜態物件里程計 =====
//
// v6 架構：
//   1. YOLOv8n (ONNX Runtime Web) — 物件偵測
//   2. MiDaS Small (ONNX) — 深度估測輔助前車判斷（選用）
//   3. COCO-SSD fallback — ONNX 模型檔不存在時自動降級
//   4. 狀態機 + IoU 追蹤 + EMA 平滑
//   5. GPS 測速 — 自車運動偵測

// =============================================
// DOM
// =============================================
const video = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const alertsEl = document.getElementById('alerts');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const toggleBtn = document.getElementById('toggle-btn');
const pipBtn = document.getElementById('pip-btn');
const flash = document.getElementById('flash-overlay');

// =============================================
// 常數
// =============================================
// 所有判定統一每 100ms 跑一次（YOLO / MiDaS / UFLD / 光流）
const DETECT_INTERVAL = 100;

// YOLOv8 推論
const YOLO_INPUT_SIZE = 640;
const YOLO_CONF_THRESHOLD = 0.25;
const YOLO_IOU_THRESHOLD = 0.45;

// MiDaS
const MIDAS_INPUT_SIZE = 256;

// UFLD (Ultra-Fast-Lane-Detection)
const UFLD_INPUT_W = 800;
const UFLD_INPUT_H = 288;
const UFLD_NUM_GRIDDING = 100;   // TuSimple griding_num
const UFLD_NUM_CLS = 56;         // row anchors 數量
const UFLD_NUM_LANES = 4;        // 最多 4 條車道線
const UFLD_RUN_EVERY = 1;        // 每 tick 跑一次
// TuSimple row anchors (原圖 720p 下的 y 像素)
const UFLD_ROW_ANCHORS = [
  64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
  116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160,
  164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208,
  212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256,
  260, 264, 268, 272, 276, 280, 284
];

// 狀態機門檻
const LOCK_ENTER_FRAMES = 3;
const DISAPPEAR_EXIT_FRAMES = 5;

// 前車移動判定
const MOVE_ENTER_RATIO = 0.06;
const MOVE_EXIT_RATIO = 0.025;
const MOVE_Y_ENTER = 0.025;
const MOVE_HISTORY = 12;
const MOVE_CONFIRM_FRAMES = 4;

// ---------- 光流起步判定（v2：仿射 ego-motion 補償 + 證據累積）----------
// 思路：
//   1. 鎖定前車 bbox 後，bbox 內撒前景點，bbox 外撒背景點
//   2. 用 LK 光流追蹤兩組點
//   3. 從背景點的位移用 RANSAC 估計「整張畫面」的相似變換 T
//      （平移+旋轉+尺度）— 這是手機晃動 / 相機微震的完整模型，
//      比「扣背景中位數」更穩健，能吃掉 roll、握持微震、focal-length 抖動
//   4. 把前景點的 prevPts 用 T warp → 預測位置；殘差 = 真實位移
//   5. 殘差在「以 FOE 為原點」的徑向投影：朝 FOE = 起步證據，反之 = 靠近證據
//   6. 用序列機率比檢定（SPRT）累積證據：
//        每點 sign 是一個 Bernoulli 觀測；LLR 累加；
//        證據強 → 1 tick 就觸發；雜訊 → 自動多看幾幀；
//        參數只有「可容忍誤警率 α」，不再是「票差≥M、連續≥N tick」這種拍腦袋數字
const OF_MAX_FG_POINTS   = 60;
const OF_MAX_BG_POINTS   = 80;       // 背景點多撒一點，仿射估計才穩
const OF_RESAMPLE_EVERY  = 900;      // ms
const OF_WIN_SIZE        = 15;
const OF_PYR_LEVELS      = 3;
const OF_MIN_FG_ALIVE    = 10;
const OF_MIN_BG_FOR_AFFINE = 8;      // 仿射估計最少需要的背景點
const OF_BBOX_SHRINK     = 0.12;
const OF_INPUT_W         = 320;
const OF_INPUT_H         = 180;
const OF_FOE_Y_RATIO     = 0.45;
const OF_DEPTH_MAD_K     = 2.5;

// SPRT 證據累積參數
//   H0: p = 0.5 (純雜訊，朝/離 FOE 機率各半)
//   H1: p = OF_SPRT_P1 (有起步 → 多數朝 FOE)
//   α  = false alarm 機率（誤警率）
//   β  = miss 機率（漏報率）
// 觸發門檻 A = ln((1-β)/α)；重置門檻 B = ln(β/(1-α))
// 每點貢獻 ±OF_LLR_UNIT；LLR 在中間段持續觀察
const OF_SPRT_P1     = 0.70;
const OF_SPRT_ALPHA  = 0.01;
const OF_SPRT_BETA   = 0.05;
const OF_RADIAL_DOMINANCE = 1.0;     // |radial| 必須 ≥ |tangential| 才算「徑向票」
const OF_LLR_DECAY   = 0.85;         // 每 tick 對舊 LLR 的折扣（避免長期累積偏置）

// EMA（更低 = 更平滑，抗抖動）
const EMA_ALPHA = 0.18;

// IoU
const IOU_LOCK_MATCH = 0.15;
const IOU_CANDIDATE = 0.25;

// 自車運動 — GPS 測速
// GPS：navigator.geolocation 提供 speed (m/s)
const GPS_MOVE_SPEED = 1.5;          // > 1.5 m/s (~5.4 km/h) → 移動
const GPS_STILL_SPEED = 0.5;         // < 0.5 m/s (~1.8 km/h) → 靜止

// 紅綠燈
const LIGHT_CONFIRM_FRAMES = 2;

// 其他
const ALERT_COOLDOWN = 4000;
const SCREEN_OFF_TIMEOUT = 5 * 60 * 1000;

// YOLOv8 COCO 類別 ID
const VEHICLE_CLASSES = new Set([2, 5, 7]);  // car, bus, truck
const LIGHT_CLASS = 9;                        // traffic light

// =============================================
// 狀態機
// =============================================
const STATE = {
  IDLE: 'idle',
  LOCKING: 'locking',
  TRACKING: 'tracking',
  DEPARTING: 'departing',
};

let sysState = STATE.IDLE;

// =============================================
// 追蹤資料
// =============================================
let running = false;
let debugMode = false;
let wakeLock = null;
let lastAlertTime = { move: 0, green: 0 };

// 功能開關（可由 UI 分別啟用）
// 規則：
//   - 前車起步偵測：無論紅綠燈與否皆通知
//   - 紅綠燈偵測：無論有無前車皆通知紅→綠
let enableCarDepart = true;
let enableTrafficLight = true;

// 模型
let yoloSession = null;
let midasSession = null;
let ufldSession = null;
let cocoModel = null;
let useOnnx = false;
let useMidas = false;
let useUfld = false;

// 前車
let smoothedBbox = null;
let lockFrameCount = 0;
let disappearFrameCount = 0;
let moveConfirmCount = 0;
let bboxHistory = [];
let lockCandidate = null;
let outOfLaneCount = 0;              // 連續多少 tick 偏離自車道

// 紅綠燈
let lockedLight = null;
let trafficLightState = 'unknown';
let lightConfirmCount = 0;

// 深度圖
let lastDepthMap = null;
let depthFrameCounter = 0;
const DEPTH_RUN_EVERY = 1;  // 每 tick 跑一次

// 自車運動
let egoMoving = false;
let gpsWatchId = null;               // GPS watchPosition ID
let lastGpsSpeed = null;             // 最新 GPS 速度 (m/s)，null = 無 GPS
let lastGpsTime = 0;                 // 最後收到 GPS 的時間

// 車道線 (UFLD)
let lastLanes = null;                // 最近一次車道線結果 [{x, y}[], ...]
let lastEgoLane = null;              // {leftX(y), rightX(y)} 自車車道邊界函數
let ufldFrameCounter = 0;

// ---------- 光流 (OpenCV.js) ----------
let cvReady = false;                 // OpenCV runtime 載入完成
let ofPrevGray = null;               // 上一幀灰階 (cv.Mat, OF_INPUT_W × OF_INPUT_H)
let ofFgPts = null;                  // 前景特徵點 cv.Mat (N×1, CV_32FC2)
let ofBgPts = null;                  // 背景特徵點 cv.Mat (N×1, CV_32FC2)
let ofFgAliveMask = null;            // 前景點存活 Uint8Array
let ofBgAliveMask = null;
let ofLastResampleTime = 0;
let ofLLR = 0;                       // SPRT 累積對數似然比；> A 觸發、< B 重置
let ofLastTickTime = 0;
let ofBboxAtSample = null;           // 撒點時的 bbox（原始座標）
// 光流用 canvas（降採樣）
const ofCanvas = document.createElement('canvas');
const ofCtx = ofCanvas.getContext('2d', { willReadFrequently: true });

// 各模型下次可執行時間（節流）
let nextYoloTime = 0;
let nextDepthTime = 0;
let nextUfldTime = 0;
let lastPredictions = [];            // 最近一次 YOLO 結果（給繪製用）

// 螢幕
let screenOff = false;
let screenOffTimer = null;

// 重用 canvas
const tmpCanvas = document.createElement('canvas');
const tmpCtx = tmpCanvas.getContext('2d');
const preCanvas = document.createElement('canvas');
const preCtx = preCanvas.getContext('2d', { willReadFrequently: true });

// =============================================
// 音效
// =============================================
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function beep(freq, dur, vol = 0.35) {
  const o = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  o.connect(g); g.connect(audioCtx.destination);
  o.frequency.value = freq; o.type = 'sine';
  g.gain.value = vol;
  o.start();
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
  o.stop(audioCtx.currentTime + dur);
}

function alertSound(type) {
  if (type === 'move') {
    beep(880, 0.15); setTimeout(() => beep(1100, 0.15), 170);
  } else {
    beep(660, 0.12); setTimeout(() => beep(880, 0.12), 140);
    setTimeout(() => beep(1100, 0.15), 280);
  }
}

function flashScreen(type) {
  flash.className = type === 'move' ? 'move-flash' : 'green-flash';
  flash.classList.add('active');
  setTimeout(() => flash.classList.remove('active'), 400);
}

function vibrate(p) { if (navigator.vibrate) navigator.vibrate(p); }

function tryAlert(type, text) {
  const now = Date.now();
  if (now - lastAlertTime[type] < ALERT_COOLDOWN) return null;
  alertSound(type);
  flashScreen(type);
  vibrate(type === 'green' ? [100, 50, 100, 50, 100] : [100, 50, 100]);
  lastAlertTime[type] = now;
  return { type, text };
}

// =============================================
// 相機 / Wake Lock / PiP
// =============================================
async function startCamera() {
  const s = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = s;
  await video.play();
}

function stopCamera() {
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  video.srcObject = null;
}

async function requestWakeLock() {
  try {
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener('release', () => { wakeLock = null; });
  } catch {}
}

async function togglePiP() {
  try {
    if (document.pictureInPictureElement) await document.exitPictureInPicture();
    else if (video.requestPictureInPicture) await video.requestPictureInPicture();  } catch {}
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

// ---------- OpenCV.js 載入 ----------
// opencv.js 約 ~10MB，首次載入慢，Service Worker 會幫忙快取。
// 載入後 window.cv 是一個 Module，需要等 onRuntimeInitialized。
async function loadOpenCv() {
  if (cvReady) return true;
  if (typeof cv === 'undefined') {
    try {
      await loadScript('https://docs.opencv.org/4.9.0/opencv.js');
    } catch (e) {
      console.warn('[A-Eye] OpenCV.js 載入失敗:', e.message);
      return false;
    }
  }
  // 等待 runtime 初始化
  await new Promise((resolve) => {
    if (cv && cv.Mat) { resolve(); return; }
    // opencv.js 4.x 載完後可能還在編譯 WASM
    const check = () => {
      if (cv && cv.Mat) resolve();
      else setTimeout(check, 50);
    };
    // 有些版本支援 onRuntimeInitialized
    if (cv && typeof cv.then === 'function') {
      cv.then(() => resolve());
    } else if (cv) {
      cv.onRuntimeInitialized = () => resolve();
      check();
    } else {
      setTimeout(check, 50);
    }
  });
  cvReady = !!(cv && cv.Mat);
  if (cvReady) console.log('[A-Eye] OpenCV.js ready');
  return cvReady;
}

// =============================================
// 載入模型
// =============================================
async function loadModels() {
  statusText.textContent = '載入 AI 模型中...';

  // 嘗試載入 YOLOv8n ONNX
  if (typeof ort !== 'undefined') {
    try {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';      statusText.textContent = '載入 YOLOv8s 模型...';
      // 優先用 yolov8s（精準度↑）；找不到再退回 yolov8n
      try {
        yoloSession = await ort.InferenceSession.create('./models/yolov8s.onnx', {
          executionProviders: ['webgl', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        console.log('[A-Eye] YOLOv8s ONNX 載入成功');
      } catch (e8s) {
        console.warn('[A-Eye] yolov8s.onnx 不存在，退回 yolov8n:', e8s.message);
        yoloSession = await ort.InferenceSession.create('./models/yolov8n.onnx', {
          executionProviders: ['webgl', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        console.log('[A-Eye] YOLOv8n ONNX 載入成功 (fallback)');
      }
      useOnnx = true;

      // 嘗試載入 MiDaS
      try {
        statusText.textContent = '載入深度模型...';        // MiDaS 含 int64 op，WebGL backend 不支援，只能用 wasm
        midasSession = await ort.InferenceSession.create('./models/midas_small.onnx', {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        useMidas = true;
        console.log('[A-Eye] MiDaS Small ONNX 載入成功');      } catch (e) {
        console.warn('[A-Eye] MiDaS 載入失敗（深度停用）:', e.message);
        useMidas = false;
      }

      // 嘗試載入 UFLD (Ultra-Fast-Lane-Detection)
      try {
        statusText.textContent = '載入車道線模型...';
        ufldSession = await ort.InferenceSession.create('./models/ufld_tusimple.onnx', {
          executionProviders: ['webgl', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        useUfld = true;
        console.log('[A-Eye] UFLD ONNX 載入成功');
      } catch (e) {
        console.warn('[A-Eye] UFLD 載入失敗（車道線停用）:', e.message);
        useUfld = false;
      }    } catch (e) {
      console.warn('[A-Eye] YOLOv8s 載入失敗，降級到 COCO-SSD:', e.message);
      useOnnx = false;
    }
  }// Fallback: COCO-SSD（動態載入 TF.js + COCO-SSD）
  if (!useOnnx) {
    statusText.textContent = '載入 COCO-SSD 模型...';
    try {
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
      cocoModel = await cocoSsd.load({ base: 'mobilenet_v2' });
      console.log('[A-Eye] COCO-SSD 載入成功 (fallback)');
    } catch (e2) {
      throw new Error('無法載入任何偵測模型');
    }
  }  const info = useOnnx ? `YOLOv8s${useMidas ? ' + MiDaS' : ''}${useUfld ? ' + UFLD' : ''}` : 'COCO-SSD';
  statusText.textContent = `模型: ${info}`;

  // 光流 (OpenCV.js)：非同步載入，不阻斷偵測啟動
  statusText.textContent = `${info}｜載入光流...`;
  loadOpenCv().then(ok => {
    if (ok) statusText.textContent = `${info} + OF`;
  }).catch(() => { /* ignore */ });
}

// =============================================
// YOLOv8n 前處理 + 推論 + NMS
// =============================================
function yoloPreprocess(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const size = YOLO_INPUT_SIZE;

  preCanvas.width = size;
  preCanvas.height = size;

  // letterbox：保持比例，填黑邊
  const scale = Math.min(size / vw, size / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const dx = Math.round((size - nw) / 2);
  const dy = Math.round((size - nh) / 2);

  preCtx.fillStyle = '#000';
  preCtx.fillRect(0, 0, size, size);
  preCtx.drawImage(videoEl, 0, 0, vw, vh, dx, dy, nw, nh);

  const imgData = preCtx.getImageData(0, 0, size, size);
  const pixels = imgData.data;

  // NCHW float32, 0~1
  const float32 = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = pixels[i * 4] / 255;
    float32[size * size + i] = pixels[i * 4 + 1] / 255;
    float32[2 * size * size + i] = pixels[i * 4 + 2] / 255;
  }

  return { tensor: float32, scale, dx, dy };
}

async function yoloDetect(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const { tensor, scale, dx, dy } = yoloPreprocess(videoEl);

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
  const feeds = {};
  feeds[yoloSession.inputNames[0]] = inputTensor;

  const results = await yoloSession.run(feeds);
  const output = results[yoloSession.outputNames[0]];

  // YOLOv8 output: [1, 84, 8400] → [8400, 84]
  const data = output.data;
  const numClasses = 80;
  const numDet = output.dims[2];

  const boxes = [];
  for (let i = 0; i < numDet; i++) {
    const cx = data[0 * numDet + i];
    const cy = data[1 * numDet + i];
    const w  = data[2 * numDet + i];
    const h  = data[3 * numDet + i];

    let maxScore = 0, maxClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numDet + i];
      if (score > maxScore) { maxScore = score; maxClass = c; }
    }

    if (maxScore < YOLO_CONF_THRESHOLD) continue;

    // letterbox → 原始座標
    const x1 = (cx - w / 2 - dx) / scale;
    const y1 = (cy - h / 2 - dy) / scale;
    const bw = w / scale;
    const bh = h / scale;

    const clampX = Math.max(0, Math.min(x1, vw));
    const clampY = Math.max(0, Math.min(y1, vh));
    const clampW = Math.min(bw, vw - clampX);
    const clampH = Math.min(bh, vh - clampY);

    if (clampW > 0 && clampH > 0) {
      boxes.push({
        bbox: [clampX, clampY, clampW, clampH],
        class: YOLO_CLASSES[maxClass],
        classId: maxClass,
        score: maxScore,
      });
    }
  }

  return nms(boxes, YOLO_IOU_THRESHOLD);
}

function nms(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  const suppressed = new Set();

  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed.has(j)) continue;
      if (boxes[i].classId !== boxes[j].classId) continue;
      if (calcIoUArray(boxes[i].bbox, boxes[j].bbox) > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}

function calcIoUArray(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

// =============================================
// MiDaS 深度估測
// =============================================
function midasPreprocess(videoEl) {
  const size = MIDAS_INPUT_SIZE;
  preCanvas.width = size;
  preCanvas.height = size;
  preCtx.drawImage(videoEl, 0, 0, size, size);
  const pixels = preCtx.getImageData(0, 0, size, size).data;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const float32 = new Float32Array(3 * size * size);

  for (let i = 0; i < size * size; i++) {
    float32[i]                    = (pixels[i * 4]     / 255 - mean[0]) / std[0];
    float32[size * size + i]      = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
    float32[2 * size * size + i]  = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return float32;
}

async function midasEstimateDepth(videoEl) {
  if (!midasSession) return null;
  const tensor = midasPreprocess(videoEl);
  const input = new ort.Tensor('float32', tensor, [1, 3, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE]);
  const feeds = {};
  feeds[midasSession.inputNames[0]] = input;
  const results = await midasSession.run(feeds);
  return results[midasSession.outputNames[0]].data;
}

function getDepthForBbox(depthMap, bbox, vw, vh) {
  if (!depthMap) return 0;
  const [bx, by, bw, bh] = bbox;
  const size = MIDAS_INPUT_SIZE;
  const x1 = Math.floor(bx / vw * size);
  const y1 = Math.floor(by / vh * size);
  const x2 = Math.ceil((bx + bw) / vw * size);
  const y2 = Math.ceil((by + bh) / vh * size);

  let sum = 0, count = 0;
  for (let y = Math.max(0, y1); y < Math.min(size, y2); y++) {
    for (let x = Math.max(0, x1); x < Math.min(size, x2); x++) {
      sum += depthMap[y * size + x];
      count++;
    }
  }  return count > 0 ? sum / count : 0;
}

// =============================================
// UFLD 車道線偵測
// =============================================
function ufldPreprocess(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  preCanvas.width = UFLD_INPUT_W;
  preCanvas.height = UFLD_INPUT_H;
  preCtx.drawImage(videoEl, 0, 0, vw, vh, 0, 0, UFLD_INPUT_W, UFLD_INPUT_H);
  const pixels = preCtx.getImageData(0, 0, UFLD_INPUT_W, UFLD_INPUT_H).data;

  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const total = UFLD_INPUT_W * UFLD_INPUT_H;
  const float32 = new Float32Array(3 * total);
  for (let i = 0; i < total; i++) {
    float32[i]             = (pixels[i * 4]     / 255 - mean[0]) / std[0];
    float32[total + i]     = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
    float32[2 * total + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return float32;
}

async function ufldDetectLanes(videoEl) {
  if (!ufldSession) return null;
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const tensor = ufldPreprocess(videoEl);
  const input = new ort.Tensor('float32', tensor, [1, 3, UFLD_INPUT_H, UFLD_INPUT_W]);
  const feeds = {};
  feeds[ufldSession.inputNames[0]] = input;
  const results = await ufldSession.run(feeds);
  const output = results[ufldSession.outputNames[0]].data;

  // output shape: [1, (UFLD_NUM_GRIDDING+1) * UFLD_NUM_CLS * UFLD_NUM_LANES]
  // reshape → [UFLD_NUM_LANES][UFLD_NUM_CLS][UFLD_NUM_GRIDDING+1]
  const G = UFLD_NUM_GRIDDING, C = UFLD_NUM_CLS, L = UFLD_NUM_LANES;
  const lanes = [];

  for (let lane = 0; lane < L; lane++) {
    const points = [];
    for (let cls = 0; cls < C; cls++) {
      // softmax 取 argmax
      const offset = (lane * C + cls) * (G + 1);
      let maxVal = -Infinity, maxIdx = G; // G = "不存在"
      for (let g = 0; g <= G; g++) {
        if (output[offset + g] > maxVal) {
          maxVal = output[offset + g];
          maxIdx = g;
        }
      }
      if (maxIdx === G) continue; // 該 row 無車道線

      // gridding index → 原始影像 x 座標
      const x = (maxIdx + 0.5) / G * vw;
      // row anchor → 原始影像 y 座標 (row anchors 基於 720p)
      const y = UFLD_ROW_ANCHORS[cls] / 720 * vh;
      points.push({ x, y });
    }
    if (points.length >= 2) lanes.push(points);
  }
  return lanes;
}

/**
 * 從偵測到的車道線中判定自車車道 (ego-lane) 的左右邊界。
 * 策略：取在畫面底部最接近中心的左側線 & 右側線。
 *
 * @returns {leftLane, rightLane} 各為 [{x,y},...] 或 null
 */
function findEgoLane(lanes, vw, vh) {
  if (!lanes || lanes.length < 2) return null;

  const centerX = vw / 2;
  const bottomY = vh * 0.9;

  // 對每條線，取其底部 (最大 y) 的 x 值
  const lanesWithBottomX = lanes.map(pts => {
    // 取最靠近 bottomY 的點
    let best = pts[pts.length - 1]; // 最後一個通常最低
    for (const p of pts) {
      if (Math.abs(p.y - bottomY) < Math.abs(best.y - bottomY)) best = p;
    }
    return { pts, bottomX: best.x };
  });

  // 分左 (bottomX < center) 右 (bottomX > center)
  let bestLeft = null, bestRight = null;
  let bestLeftDist = Infinity, bestRightDist = Infinity;

  for (const l of lanesWithBottomX) {
    const dist = Math.abs(l.bottomX - centerX);
    if (l.bottomX <= centerX && dist < bestLeftDist) {
      bestLeftDist = dist;
      bestLeft = l.pts;
    }
    if (l.bottomX > centerX && dist < bestRightDist) {
      bestRightDist = dist;
      bestRight = l.pts;
    }
  }

  if (!bestLeft || !bestRight) return null;
  return { leftLane: bestLeft, rightLane: bestRight };
}

/**
 * 給定 y 座標，由車道線點集用線性內插算出 x 座標
 */
function laneXatY(lanePoints, y) {
  if (!lanePoints || lanePoints.length === 0) return null;
  // 找包夾 y 的兩個點
  // lanePoints 已按 y 排序（row anchor 從上到下）
  if (y <= lanePoints[0].y) return lanePoints[0].x;
  if (y >= lanePoints[lanePoints.length - 1].y) return lanePoints[lanePoints.length - 1].x;

  for (let i = 0; i < lanePoints.length - 1; i++) {
    const p0 = lanePoints[i], p1 = lanePoints[i + 1];
    if (y >= p0.y && y <= p1.y) {
      const t = (y - p0.y) / (p1.y - p0.y);
      return p0.x + t * (p1.x - p0.x);
    }
  }
  return lanePoints[lanePoints.length - 1].x;
}

/**
 * 判斷車輛 bbox 是否在自車車道內
 * @returns 0~1 的權重，1 = 完全在車道內，0 = 完全在外
 */
function egoLaneWeight(bbox, egoLane, vh) {
  if (!egoLane) return null; // 無車道資訊，fallback

  const [bx, by, bw, bh] = bbox;
  const carCenterX = bx + bw / 2;
  const carBottomY = by + bh;

  const leftX = laneXatY(egoLane.leftLane, carBottomY);
  const rightX = laneXatY(egoLane.rightLane, carBottomY);
  if (leftX === null || rightX === null) return null;

  const laneWidth = rightX - leftX;
  if (laneWidth <= 0) return null;

  // 車輛中心在車道內的程度
  if (carCenterX >= leftX && carCenterX <= rightX) {
    return 1.0;
  }
  // 在車道外，根據距離衰減
  const distOutside = carCenterX < leftX
    ? (leftX - carCenterX) : (carCenterX - rightX);
  const decay = Math.exp(-2 * distOutside / laneWidth);
  return decay;
}

// =============================================
// 統一偵測介面
// =============================================
async function detectObjects(videoEl) {
  if (useOnnx && yoloSession) {
    return await yoloDetect(videoEl);
  }
  if (cocoModel) {
    const preds = await cocoModel.detect(videoEl, 40, 0.15);
    return preds.map(p => ({
      bbox: p.bbox,
      class: p.class,
      classId: YOLO_CLASSES.indexOf(p.class),
      score: p.score,
    }));
  }
  return [];
}

// =============================================
// 工具函數
// =============================================
function calcIoU(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

function emaSmooth(prev, curr, alpha) {
  if (!prev) return { ...curr };
  return {
    x: prev.x * (1 - alpha) + curr.x * alpha,
    y: prev.y * (1 - alpha) + curr.y * alpha,
    w: prev.w * (1 - alpha) + curr.w * alpha,
    h: prev.h * (1 - alpha) + curr.h * alpha,
  };
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// =============================================
// 光流（OpenCV.js）— 前車起步判定
// =============================================
// 只判定「方向符號」與「多數決」，完全不用位移大小閾值/百分比。

// 把 video 降採樣畫到 ofCanvas，回傳灰階 cv.Mat（需手動 delete）
function ofCaptureGray() {
  if (!cvReady) return null;
  ofCanvas.width = OF_INPUT_W;
  ofCanvas.height = OF_INPUT_H;
  ofCtx.drawImage(video, 0, 0, OF_INPUT_W, OF_INPUT_H);
  const src = cv.imread(ofCanvas);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  src.delete();
  return gray;
}

// 原始影像 bbox → 光流低解析度座標 bbox（內縮 OF_BBOX_SHRINK）
function ofBboxToLowRes(bboxRaw, vw, vh) {
  const sx = OF_INPUT_W / vw;
  const sy = OF_INPUT_H / vh;
  let x = bboxRaw.x * sx;
  let y = bboxRaw.y * sy;
  let w = bboxRaw.w * sx;
  let h = bboxRaw.h * sy;
  // 內縮避開邊緣抖動
  const dx = w * OF_BBOX_SHRINK;
  const dy = h * OF_BBOX_SHRINK;
  x += dx; y += dy; w -= 2 * dx; h -= 2 * dy;
  x = Math.max(0, x); y = Math.max(0, y);
  w = Math.min(OF_INPUT_W - x, w); h = Math.min(OF_INPUT_H - y, h);
  return { x, y, w, h };
}

function ofReleaseMat(m) { if (m && !m.isDeleted()) m.delete(); }

function ofReleaseAll() {
  ofReleaseMat(ofPrevGray); ofPrevGray = null;
  ofReleaseMat(ofFgPts);    ofFgPts    = null;
  ofReleaseMat(ofBgPts);    ofBgPts    = null;
  ofBboxAtSample = null;
  ofLLR = 0;
  ofLastTickTime = 0;
  ofLastResampleTime = 0;
}

// 用 Shi-Tomasi 在 ROI 內撒特徵點，回傳 cv.Mat (N×1, CV_32FC2)
// roi = {x,y,w,h} in low-res coords；maskInvertBbox=true 表示在 bbox 外撒（背景）
function ofSamplePoints(gray, bbox, nMax, excludeBbox) {
  const mask = cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8UC1);
  if (excludeBbox) {
    // 整張圖 = 255，bbox 內填 0
    mask.setTo(new cv.Scalar(255));
    const rect = new cv.Rect(
      Math.round(bbox.x), Math.round(bbox.y),
      Math.round(bbox.w), Math.round(bbox.h)
    );
    const roi = mask.roi(rect);
    roi.setTo(new cv.Scalar(0));
    roi.delete();
    // 再去除畫面最邊緣（常有鏡頭黑邊）
    const bw = 6;
    cv.rectangle(mask, new cv.Point(0, 0), new cv.Point(gray.cols - 1, bw), new cv.Scalar(0), -1);
    cv.rectangle(mask, new cv.Point(0, gray.rows - bw), new cv.Point(gray.cols - 1, gray.rows - 1), new cv.Scalar(0), -1);
    cv.rectangle(mask, new cv.Point(0, 0), new cv.Point(bw, gray.rows - 1), new cv.Scalar(0), -1);
    cv.rectangle(mask, new cv.Point(gray.cols - bw, 0), new cv.Point(gray.cols - 1, gray.rows - 1), new cv.Scalar(0), -1);
  } else {
    // bbox 內 = 255
    const rect = new cv.Rect(
      Math.round(bbox.x), Math.round(bbox.y),
      Math.round(bbox.w), Math.round(bbox.h)
    );
    const roi = mask.roi(rect);
    roi.setTo(new cv.Scalar(255));
    roi.delete();
  }

  const corners = new cv.Mat();
  try {
    cv.goodFeaturesToTrack(gray, corners, nMax, 0.01, 5, mask, 3, false, 0.04);
  } catch (e) {
    mask.delete();
    corners.delete();
    return null;
  }
  mask.delete();
  if (corners.rows === 0) { corners.delete(); return null; }
  return corners;
}

// 用 LK 追蹤 prevPts → nextPts，回傳 {nextPts, status}
// 存活點抽出成新 cv.Mat；失敗/追丟點會被移除。
function ofTrack(prevGray, nextGray, prevPts) {
  if (!prevPts || prevPts.rows === 0) return null;
  const nextPts = new cv.Mat();
  const status  = new cv.Mat();
  const err     = new cv.Mat();
  const winSize = new cv.Size(OF_WIN_SIZE, OF_WIN_SIZE);
  const criteria = new cv.TermCriteria(
    cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 20, 0.03
  );
  try {
    cv.calcOpticalFlowPyrLK(
      prevGray, nextGray, prevPts, nextPts, status, err,
      winSize, OF_PYR_LEVELS, criteria
    );
  } catch (e) {
    nextPts.delete(); status.delete(); err.delete();
    return null;
  }

  // 收集存活點 (status == 1 且在影像內)
  const alivePrev = [], aliveNext = [];
  for (let i = 0; i < status.rows; i++) {
    if (status.data[i] !== 1) continue;
    const px = prevPts.data32F[i * 2],     py = prevPts.data32F[i * 2 + 1];
    const nx = nextPts.data32F[i * 2],     ny = nextPts.data32F[i * 2 + 1];
    if (nx < 0 || ny < 0 || nx >= OF_INPUT_W || ny >= OF_INPUT_H) continue;
    alivePrev.push(px, py);
    aliveNext.push(nx, ny);
  }
  nextPts.delete(); status.delete(); err.delete();

  if (alivePrev.length === 0) return null;
  const N = alivePrev.length / 2;
  const newPrev = cv.matFromArray(N, 1, cv.CV_32FC2, alivePrev);
  const newNext = cv.matFromArray(N, 1, cv.CV_32FC2, aliveNext);
  return { prevPts: newPrev, nextPts: newNext };
}

// 在給定的 bbox (low-res) 重新撒前景 + 背景點
function ofResample(gray, bboxLow) {
  ofReleaseMat(ofFgPts); ofFgPts = null;
  ofReleaseMat(ofBgPts); ofBgPts = null;
  ofFgPts = ofSamplePoints(gray, bboxLow, OF_MAX_FG_POINTS, false);
  // 用 MiDaS 深度把前景點中「不在車體上」的離群點剃除
  ofFgPts = ofFilterPointsByDepth(ofFgPts);
  ofBgPts = ofSamplePoints(gray, bboxLow, OF_MAX_BG_POINTS, true);
  ofBboxAtSample = { ...bboxLow };
  ofLastResampleTime = Date.now();
}

// ---------- MiDaS 深度過濾（對一組低解析度特徵點）----------
// 原理：同一台車上的點，MiDaS 深度值應接近；用「中位數 ± K·MAD」判離群。
// 完全不使用固定像素/百分比閾值，K 是統計意義上的倍數。
// 輸入 / 輸出均為 cv.Mat (N×1, CV_32FC2, low-res 座標)，若無深度圖則原樣回傳。
function ofFilterPointsByDepth(ptsMat) {
  if (!ptsMat || ptsMat.rows === 0) return ptsMat;
  if (!lastDepthMap) return ptsMat;  // 沒深度圖就不過濾

  const N = ptsMat.rows;
  const depths = new Array(N);
  // low-res (OF_INPUT_W × OF_INPUT_H) → MiDaS (MIDAS_INPUT_SIZE × MIDAS_INPUT_SIZE)
  const sx = MIDAS_INPUT_SIZE / OF_INPUT_W;
  const sy = MIDAS_INPUT_SIZE / OF_INPUT_H;
  for (let i = 0; i < N; i++) {
    const px = ptsMat.data32F[i * 2];
    const py = ptsMat.data32F[i * 2 + 1];
    let mx = Math.floor(px * sx);
    let my = Math.floor(py * sy);
    if (mx < 0) mx = 0; else if (mx >= MIDAS_INPUT_SIZE) mx = MIDAS_INPUT_SIZE - 1;
    if (my < 0) my = 0; else if (my >= MIDAS_INPUT_SIZE) my = MIDAS_INPUT_SIZE - 1;
    depths[i] = lastDepthMap[my * MIDAS_INPUT_SIZE + mx];
  }

  const dMed = median(depths);
  const mad = median(depths.map(d => Math.abs(d - dMed))) || 1e-6;
  const thresh = OF_DEPTH_MAD_K * mad;

  const kept = [];
  for (let i = 0; i < N; i++) {
    if (Math.abs(depths[i] - dMed) <= thresh) {
      kept.push(ptsMat.data32F[i * 2], ptsMat.data32F[i * 2 + 1]);
    }
  }
  ptsMat.delete();
  if (kept.length === 0) return null;
  return cv.matFromArray(kept.length / 2, 1, cv.CV_32FC2, kept);
}

// 核心：每 tick 呼叫，回傳 { departed: bool, info: string }
// bboxRaw: 當前前車 bbox（原始座標），可能為 null（無前車）
function ofTick(bboxRaw, vw, vh) {
  if (!cvReady) return { departed: false, info: 'cv-not-ready' };

  // 無前車 or 自車行駛 → 重置
  if (!bboxRaw || egoMoving) {
    if (ofPrevGray || ofFgPts) ofReleaseAll();
    return { departed: false, info: 'no-target' };
  }

  const now = Date.now();
  const gray = ofCaptureGray();
  if (!gray) return { departed: false, info: 'capture-fail' };

  const bboxLow = ofBboxToLowRes(bboxRaw, vw, vh);
  if (bboxLow.w < 10 || bboxLow.h < 10) {
    gray.delete();
    return { departed: false, info: 'bbox-too-small' };
  }
  // 第一次呼叫：撒點即可
  if (!ofPrevGray) {
    ofResample(gray, bboxLow);
    ofPrevGray = gray;
    ofLLR = 0;
    return { departed: false, info: 'init' };
  }

  // 若 bbox 跳動太大（中心偏移超過 bbox 寬度的一半）→ 重撒
  if (ofBboxAtSample) {
    const cx0 = ofBboxAtSample.x + ofBboxAtSample.w / 2;
    const cy0 = ofBboxAtSample.y + ofBboxAtSample.h / 2;
    const cx1 = bboxLow.x + bboxLow.w / 2;
    const cy1 = bboxLow.y + bboxLow.h / 2;
    const dCenter = Math.hypot(cx1 - cx0, cy1 - cy0);
    if (dCenter > Math.max(bboxLow.w, bboxLow.h) * 0.5) {
      ofResample(gray, bboxLow);
      ofReleaseMat(ofPrevGray); ofPrevGray = gray;
      ofLLR = 0;
      return { departed: false, info: 'bbox-jump' };
    }
  }

  // 追蹤前景 + 背景
  const fgRes = ofTrack(ofPrevGray, gray, ofFgPts);
  const bgRes = ofTrack(ofPrevGray, gray, ofBgPts);

  let departed = false;
  let info = 'no-vote';

  if (fgRes && bgRes
      && fgRes.nextPts.rows >= OF_MIN_FG_ALIVE
      && bgRes.nextPts.rows >= OF_MIN_BG_FOR_AFFINE) {

    // ---- (A) 用背景點估計 ego-motion 仿射變換 (RANSAC) ----
    // 平移+旋轉+尺度（4 自由度），補償手機晃動 / roll / 焦距微震
    let M = null;
    try {
      M = cv.estimateAffinePartial2D(
        bgRes.prevPts, bgRes.nextPts,
        new cv.Mat(), cv.RANSAC, 2.0, 2000, 0.99, 10
      );
    } catch (e) { M = null; }

    // 從 M 取出 [a, b, tx; -b, a, ty]；若 M 不存在則退回單位矩陣（無補償）
    let a = 1, b = 0, tx = 0, ty = 0;
    if (M && !M.empty() && M.rows === 2 && M.cols === 3) {
      a  = M.doubleAt(0, 0); b  = M.doubleAt(0, 1); tx = M.doubleAt(0, 2);
      ty = M.doubleAt(1, 2);
    }
    if (M) M.delete();

    // ---- (B) FOE：畫面中央偏上 ----
    const foeX = OF_INPUT_W * 0.5;
    const foeY = OF_INPUT_H * OF_FOE_Y_RATIO;

    // ---- (C) 對每個前景點：用 ego-motion warp 預測位置，殘差取徑向 ----
    const fgN = fgRes.nextPts.rows;
    let voteDepart = 0, voteApproach = 0, used = 0;
    for (let i = 0; i < fgN; i++) {
      const px = fgRes.prevPts.data32F[i * 2];
      const py = fgRes.prevPts.data32F[i * 2 + 1];
      const nx = fgRes.nextPts.data32F[i * 2];
      const ny = fgRes.nextPts.data32F[i * 2 + 1];
      // ego-motion 預測位置：T(prev)
      const ex = a * px + b * py + tx;
      const ey = -b * px + a * py + ty;
      // 殘差 = 真實位置 − ego-motion 預測（吃掉相機晃動後的真位移）
      const dx = nx - ex;
      const dy = ny - ey;

      // 以該點 prev 位置到 FOE 的徑向
      const rx = px - foeX, ry = py - foeY;
      const rn = Math.hypot(rx, ry);
      if (rn < 1e-3) continue;
      const radial = (dx * rx + dy * ry) / rn;
      const tx_ = -ry / rn, ty_ = rx / rn;
      const tangential = dx * tx_ + dy * ty_;

      // 只計入「徑向佔主導」的點（排除純橫向/雜訊）
      if (Math.abs(radial) < OF_RADIAL_DOMINANCE * Math.abs(tangential)) continue;
      used++;
      if (radial < 0) voteDepart++;
      else voteApproach++;
    }

    // ---- (D) 序列機率比檢定 (SPRT) 累積證據 ----
    // 每個徑向票的 sign 是一個 Bernoulli 樣本：
    //   H0: P(朝FOE) = 0.5（雜訊）
    //   H1: P(朝FOE) = OF_SPRT_P1（起步）
    // 單點 LLR：
    //   朝FOE 一點 → +ln(P1 / 0.5)
    //   離FOE 一點 → +ln((1-P1) / 0.5)
    const llrDepart   = Math.log(OF_SPRT_P1   / 0.5);
    const llrApproach = Math.log((1 - OF_SPRT_P1) / 0.5);
    const A = Math.log((1 - OF_SPRT_BETA) / OF_SPRT_ALPHA);   // 上門檻 → 觸發
    const B = Math.log(OF_SPRT_BETA / (1 - OF_SPRT_ALPHA));   // 下門檻 → 重置

    if (used > 0) {
      // 折扣舊證據（避免長期偏置）
      ofLLR = ofLLR * OF_LLR_DECAY
            + voteDepart * llrDepart
            + voteApproach * llrApproach;
    } else {
      ofLLR *= OF_LLR_DECAY;
    }

    info = `fg:${fgN} bg:${bgRes.nextPts.rows} u:${used} `
         + `d:${voteDepart} a:${voteApproach} LLR:${ofLLR.toFixed(2)}`
         + ` (A=${A.toFixed(2)})`;

    if (ofLLR >= A) {
      departed = true;
      ofLLR = 0;
    } else if (ofLLR <= B) {
      ofLLR = 0;  // 確認非起步，重置從零累積
    }
  } else {
    // 點數不足 → 證據衰退
    ofLLR *= OF_LLR_DECAY;
  }

  // 更新 prev 為當前灰階；特徵點沿用追蹤到的「新位置」以繼續追下一幀
  ofReleaseMat(ofPrevGray);
  ofPrevGray = gray;

  // 用新位置取代舊特徵點
  if (fgRes) {
    ofReleaseMat(ofFgPts);
    ofFgPts = fgRes.nextPts;
    ofReleaseMat(fgRes.prevPts);
  }
  if (bgRes) {
    ofReleaseMat(ofBgPts);
    ofBgPts = bgRes.nextPts;
    ofReleaseMat(bgRes.prevPts);
  }

  // 存活點太少 或 超時 → 重撒
  const fgAlive = ofFgPts ? ofFgPts.rows : 0;
  const needResample = fgAlive < OF_MIN_FG_ALIVE
    || (now - ofLastResampleTime) > OF_RESAMPLE_EVERY;
  if (needResample) {
    ofResample(ofPrevGray, bboxLow);
  }

  ofLastTickTime = now;
  return { departed, info };
}

// =============================================
// 自車運動偵測 — GPS 測速
// =============================================
// GPS 提供 speed (m/s)，搭配遲滯門檻判斷移動/靜止。

// --- GPS 測速 ---
function startGps() {
  if (gpsWatchId !== null) return;
  if (!navigator.geolocation) {
    console.warn('[A-Eye] GPS 不支援');
    return;
  }
  gpsWatchId = navigator.geolocation.watchPosition(
    (pos) => {
      // speed: m/s, 可能為 null（無法計算時）
      lastGpsSpeed = (pos.coords.speed !== null && pos.coords.speed >= 0)
        ? pos.coords.speed : null;
      lastGpsTime = Date.now();
      // 更新 UI 速度顯示
      const speedEl = document.getElementById('speed-display');
      if (speedEl) {
        speedEl.style.display = '';
        speedEl.textContent = lastGpsSpeed !== null
          ? `${Math.round(lastGpsSpeed * 3.6)} km/h` : '-- km/h';
      }
    },
    (err) => {
      console.warn('[A-Eye] GPS 錯誤:', err.message);
      lastGpsSpeed = null;
    },
    { enableHighAccuracy: true, maximumAge: 2000, timeout: 5000 }
  );
  console.log('[A-Eye] GPS 監聽已啟動');
}

function stopGps() {
  if (gpsWatchId !== null) {
    navigator.geolocation.clearWatch(gpsWatchId);
    gpsWatchId = null;
  }  lastGpsSpeed = null;
  lastGpsTime = 0;
  const speedEl = document.getElementById('speed-display');
  if (speedEl) speedEl.style.display = 'none';
}

function detectEgoMotionByGps() {
  // GPS 資料超過 5 秒沒更新 → 視為不可用
  if (lastGpsSpeed === null || (Date.now() - lastGpsTime) > 5000) return null;
  return lastGpsSpeed;
}

// --- GPS 決策 ---
function updateEgoMotion() {
  const gpsSpeed = detectEgoMotionByGps();
  if (gpsSpeed === null) return;  // GPS 不可用，維持上一狀態
  egoMoving = egoMoving
    ? gpsSpeed > GPS_STILL_SPEED
    : gpsSpeed > GPS_MOVE_SPEED;
}

// =============================================
// 前車篩選 — 面積 + 深度 + 車道線加權
// =============================================

/**
 * 硬性過濾：bbox 是否有資格當「前車」候選
 * 規則：
 *   1. bbox 必須在畫面下半部（避免遠方/天空/招牌誤偵測）
 *   2. bbox 面積佔比 ≥ 0.5%（避免太遠的小車）
 *   3. 若有 ego-lane → bbox 底部中心必須在自車道內（laneW > 0）
 *      若無 ego-lane → 退回中央 60% 區域（軟性備援）
 * 這些是幾何 / 物理上必要的硬閘，不是憑感覺的閾值。
 */
function isFrontCarCandidate(bbox, vw, vh) {
  const [x, y, w, h] = bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;

  // (1) 中心點 y 必須在畫面下半（前車一定在地平線下方）
  if (cy / vh < 0.4) return false;

  // (2) 面積太小（過遠）→ 排除
  const areaRatio = (w * h) / (vw * vh);
  if (areaRatio < 0.005) return false;

  // (3) 車道硬閘
  if (lastEgoLane) {
    const lw = egoLaneWeight(bbox, lastEgoLane, vh);
    // 在車道內 (lw === 1.0) 或邊緣外少量 (lw > ~0.4)：放行
    // 偏離車道太多 → 排除（大概率是旁車道車）
    if (lw === null || lw < 0.4) return false;
  } else {
    // 無車道線：退回中央 60% 區域
    const cxNorm = cx / vw;
    if (cxNorm < 0.2 || cxNorm > 0.8) return false;
  }
  return true;
}

function scoreFrontCar(bbox, vw, vh, depthMap) {
  const [x, y, w, h] = bbox;
  const areaScore = (w * h) / (vw * vh);

  // --- 車道線加權 (UFLD ego-lane) ---
  // 如果有車道線資訊，用車道邊界判斷；否則 fallback 到高斯中心加權
  let hWeight;
  const laneW = egoLaneWeight(bbox, lastEgoLane, vh);
  if (laneW !== null) {
    // 車道內 = 1.0，車道外指數衰減
    hWeight = laneW;
  } else {
    // Fallback: 高斯中心加權（無車道線時）
    const cx = (x + w / 2) / vw;          // 0~1, 0.5 = 正中央
    const hDev = (cx - 0.5) / 0.25;       // σ=0.25
    hWeight = Math.exp(-0.5 * hDev * hDev);
  }

  let baseScore;
  if (depthMap) {
    const depth = getDepthForBbox(depthMap, bbox, vw, vh);
    baseScore = depth * 0.7 + areaScore * 1000 * 0.3;
  } else {
    baseScore = areaScore;
  }

  return baseScore * hWeight;
}

// =============================================
// 前車追蹤 — 狀態機
// =============================================
function updateCarTracking(vehicles, vw, vh, depthMap) {
  const alerts = [];
  switch (sysState) {    case STATE.IDLE: {
      if (vehicles.length === 0) break;
      // 1) 硬性過濾：位置/大小/車道
      const cands = vehicles.filter(v => isFrontCarCandidate(v.bbox, vw, vh));
      if (cands.length === 0) break;

      // 2) 排序策略：
      //    - 有 MiDaS 深度圖：直接挑「深度最近」（深度值最大者）— 純模型輸出
      //    - 無深度圖：退回 scoreFrontCar 軟分數（面積×中心高斯）
      let best = null;
      if (depthMap) {
        let bestDepth = -Infinity;
        for (const v of cands) {
          const d = getDepthForBbox(depthMap, v.bbox, vw, vh);
          if (d > bestDepth) { bestDepth = d; best = v; }
        }
      } else {
        let bestS = 0;
        for (const v of cands) {
          const s = scoreFrontCar(v.bbox, vw, vh, depthMap);
          if (s > bestS) { bestS = s; best = v; }
        }
      }

      if (best) {
        const [x, y, w, h] = best.bbox;
        lockCandidate = { x, y, w, h };
        lockFrameCount = 1;
        sysState = STATE.LOCKING;
        alerts.push({ type: 'idle', text: '🚗 鎖定前車中...' });
      }
      break;
    }

    case STATE.LOCKING: {
      let matched = null, bestIoU = 0;
      for (const v of vehicles) {
        const [x, y, w, h] = v.bbox;
        const iou = calcIoU(lockCandidate, { x, y, w, h });
        if (iou > bestIoU) { bestIoU = iou; matched = { x, y, w, h, confidence: v.score }; }
      }
      if (matched && bestIoU > IOU_CANDIDATE) {
        lockCandidate = { x: matched.x, y: matched.y, w: matched.w, h: matched.h };
        lockFrameCount++;
        if (lockFrameCount >= LOCK_ENTER_FRAMES) {
          smoothedBbox = { ...lockCandidate };
          bboxHistory = [];
          moveConfirmCount = 0;
          disappearFrameCount = 0;
          sysState = STATE.TRACKING;
          alerts.push({ type: 'idle', text: `🚗 已鎖定前車（${Math.round(matched.confidence * 100)}%）` });
        } else {
          alerts.push({ type: 'idle', text: '🚗 鎖定前車中...' });
        }
      } else {
        lockCandidate = null;
        lockFrameCount = 0;
        sysState = STATE.IDLE;
      }
      break;
    }    case STATE.TRACKING: {
      let matched = null, bestIoU = 0;
      for (const v of vehicles) {
        const [x, y, w, h] = v.bbox;
        const iou = calcIoU(smoothedBbox, { x, y, w, h });
        if (iou > bestIoU) { bestIoU = iou; matched = { x, y, w, h, confidence: v.score }; }
      }      if (matched && bestIoU > IOU_LOCK_MATCH) {
        disappearFrameCount = 0;
        smoothedBbox = emaSmooth(smoothedBbox, matched, EMA_ALPHA);

        // 換道車保護：若已偏離自車道太久 → 視為旁車道車，丟棄不通知
        if (lastEgoLane) {
          const lw = egoLaneWeight(
            [smoothedBbox.x, smoothedBbox.y, smoothedBbox.w, smoothedBbox.h],
            lastEgoLane, vh
          );
          if (lw !== null && lw < 0.4) outOfLaneCount++;
          else outOfLaneCount = 0;
          if (outOfLaneCount >= 8) {  // 連續 ~800ms 偏離 → 換道車
            resetCarState();
            sysState = STATE.IDLE;
            alerts.push({ type: 'idle', text: '🚗 已偏離車道，重新尋找前車' });
            break;
          }
        }

        if (egoMoving) {
          alerts.push({ type: 'idle', text: '🚙 行駛中' });
        } else {
          // 起步判定改由光流 (ofTick) 處理，這裡只維持追蹤狀態
          alerts.push({ type: 'idle', text: `🚗 追蹤前車中（${Math.round(matched.confidence * 100)}%）` });
        }
      } else {
        disappearFrameCount = 1;
        sysState = STATE.DEPARTING;
        alerts.push({ type: 'idle', text: '🚗 確認前車狀態...' });
      }
      break;
    }

    case STATE.DEPARTING: {
      let matched = null, bestIoU = 0;
      for (const v of vehicles) {
        const [x, y, w, h] = v.bbox;
        const iou = calcIoU(smoothedBbox, { x, y, w, h });
        if (iou > bestIoU) { bestIoU = iou; matched = { x, y, w, h, confidence: v.score }; }
      }

      if (matched && bestIoU > IOU_LOCK_MATCH) {
        smoothedBbox = emaSmooth(smoothedBbox, matched, EMA_ALPHA);
        disappearFrameCount = 0;
        sysState = STATE.TRACKING;
        alerts.push({ type: 'idle', text: `🚗 追蹤前車中（${Math.round(matched.confidence * 100)}%）` });
      } else {
        disappearFrameCount++;
        if (disappearFrameCount >= DISAPPEAR_EXIT_FRAMES) {
          if (!egoMoving) {
            const a = tryAlert('move', '🚗 前車已駛離！');
            if (a) alerts.push(a);
          }
          resetCarState();
          sysState = STATE.IDLE;
        } else {
          alerts.push({ type: 'idle', text: '🚗 確認前車狀態...' });
        }
      }
      break;
    }
  }

  return alerts;
}

function resetCarState() {
  smoothedBbox = null;
  bboxHistory = [];
  lockCandidate = null;
  lockFrameCount = 0;
  disappearFrameCount = 0;  moveConfirmCount = 0;
  outOfLaneCount = 0;
}

// =============================================
// 紅綠燈
// =============================================
function scoreTrafficLight(bbox, vw, vh) {
  const [x, y, w, h] = bbox;
  const areaScore = (w * h) / (vw * vh);

  // --- 水平位置加權（高斯：中央=1）---
  const cx = (x + w / 2) / vw;
  const hDev = (cx - 0.5) / 0.3;        // σ=0.3（燈號比車輛允許稍寬）
  const hWeight = Math.exp(-0.5 * hDev * hDev);

  return areaScore * hWeight;
}

function findTrafficLight(lights) {
  if (lights.length === 0) return null;

  if (lockedLight) {
    let best = null, bestIoU = 0;
    for (const l of lights) {
      const [x, y, w, h] = l.bbox;
      const iou = calcIoU(lockedLight, { x, y, w, h });
      if (iou > bestIoU) { bestIoU = iou; best = l.bbox; }
    }
    if (best && bestIoU > 0.3) {
      const [x, y, w, h] = best;
      lockedLight = { x, y, w, h };
      return best;
    }
    lockedLight = null;
  }

  const vw = video.videoWidth, vh = video.videoHeight;
  let bestLight = null, bestScore = 0;
  for (const l of lights) {
    const s = scoreTrafficLight(l.bbox, vw, vh);
    if (s > bestScore) { bestScore = s; bestLight = l; }
  }
  if (!bestLight || bestScore === 0) return null;

  const [x, y, w, h] = bestLight.bbox;
  lockedLight = { x, y, w, h };
  return bestLight.bbox;
}

function analyzeTrafficLightColor(bbox) {
  const [tx, ty, tw, th] = bbox;
  tmpCanvas.width = Math.max(1, Math.round(tw));
  tmpCanvas.height = Math.max(1, Math.round(th));
  tmpCtx.drawImage(video, tx, ty, tw, th, 0, 0, tmpCanvas.width, tmpCanvas.height);
  const d = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height).data;

  let redScore = 0, greenScore = 0, bright = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (mx / 255 > 0.45 && (mx > 0 ? (mx - mn) / mx : 0) > 0.25) {
      bright++;
      if (r > 160 && r > g * 1.5 && r > b * 1.5) redScore++;
      if (g > 100 && g > r * 1.2 && g > b * 1.1) greenScore++;
    }
  }

  if (bright < 5) return 'unknown';
  if (redScore > bright * 0.15 && redScore > greenScore * 2) return 'red';
  if (greenScore > bright * 0.15 && greenScore > redScore * 2) return 'green';
  return 'unknown';
}

function updateTrafficLight(lights) {
  const alerts = [];
  if (lights.length === 0) return alerts;

  const bbox = findTrafficLight(lights);
  if (!bbox) return alerts;

  const color = analyzeTrafficLightColor(bbox);

  if (!egoMoving && color === 'green' && trafficLightState === 'red') {
    lightConfirmCount++;
    if (lightConfirmCount >= LIGHT_CONFIRM_FRAMES) {
      const a = tryAlert('green', '🟢 綠燈了！起步！');
      if (a) alerts.push(a);
      lightConfirmCount = 0;
    }
  } else if (color !== 'green' || trafficLightState !== 'red') {
    lightConfirmCount = 0;
  }

  if (color === 'red') {
    alerts.push({ type: 'red', text: '🔴 偵測到紅燈' });
  } else if (color === 'green' && !alerts.some(a => a.type === 'green')) {
    alerts.push({ type: 'green', text: '🟢 目前綠燈' });
  }

  if (color !== 'unknown') trafficLightState = color;
  return alerts;
}

// =============================================
// Overlay 繪製
// =============================================
function drawOverlay(predictions, vw, vh) {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  const sx = overlay.width / vw, sy = overlay.height / vh;
  octx.clearRect(0, 0, overlay.width, overlay.height);

  if (smoothedBbox && (sysState === STATE.TRACKING || sysState === STATE.DEPARTING)) {
    const { x, y, w, h } = smoothedBbox;
    octx.strokeStyle = '#3b82f6';
    octx.lineWidth = 2.5;
    octx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    octx.fillStyle = '#3b82f6';
    octx.font = 'bold 13px sans-serif';
    octx.fillText('前車', x * sx, y * sy - 5);
  }

  if (lockedLight) {
    const { x, y, w, h } = lockedLight;
    octx.strokeStyle = '#facc15';
    octx.lineWidth = 2;
    octx.setLineDash([6, 4]);
    octx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    octx.setLineDash([]);
    octx.fillStyle = '#facc15';
    octx.font = 'bold 13px sans-serif';    octx.fillText('🚦', x * sx, y * sy - 5);
  }

  // 車道線繪製 (UFLD)
  if (lastLanes && lastLanes.length > 0) {
    for (const lane of lastLanes) {
      if (lane.length < 2) continue;
      // 判斷是否為 ego-lane 邊界
      const isEgo = lastEgoLane &&
        (lane === lastEgoLane.leftLane || lane === lastEgoLane.rightLane);
      octx.strokeStyle = isEgo ? 'rgba(74,222,128,0.7)' : 'rgba(255,255,255,0.3)';
      octx.lineWidth = isEgo ? 3 : 1.5;
      octx.beginPath();
      octx.moveTo(lane[0].x * sx, lane[0].y * sy);
      for (let i = 1; i < lane.length; i++) {
        octx.lineTo(lane[i].x * sx, lane[i].y * sy);
      }
      octx.stroke();
    }
    // ego-lane 填色
    if (lastEgoLane) {
      octx.fillStyle = 'rgba(74,222,128,0.08)';
      octx.beginPath();
      const left = lastEgoLane.leftLane;
      const right = lastEgoLane.rightLane;
      // 左側線從上到下
      octx.moveTo(left[0].x * sx, left[0].y * sy);
      for (let i = 1; i < left.length; i++) octx.lineTo(left[i].x * sx, left[i].y * sy);
      // 右側線從下到上
      for (let i = right.length - 1; i >= 0; i--) octx.lineTo(right[i].x * sx, right[i].y * sy);
      octx.closePath();
      octx.fill();
    }
  }

  // 狀態
  octx.font = '11px sans-serif';
  octx.fillStyle = 'rgba(255,255,255,0.4)';
  const ml = useOnnx ? 'YOLO' : 'COCO';  const dl = useMidas ? '+D' : '';
  const ll = useUfld ? '+L' : '';
  const sl = egoMoving ? '🚙 行駛中' : `📡 ${sysState.toUpperCase()}`;
  octx.fillText(`${sl} [${ml}${dl}${ll}]`, 8, overlay.height - 8);

  // Debug
  if (debugMode) {
    octx.font = '10px monospace';
    for (const p of predictions) {
      const [bx, by, bw, bh] = p.bbox;
      const isV = VEHICLE_CLASSES.has(p.classId);
      const isL = p.classId === LIGHT_CLASS;
      octx.strokeStyle = isV ? 'rgba(0,255,0,0.6)' : isL ? 'rgba(255,255,0,0.6)' : 'rgba(255,0,255,0.4)';
      octx.lineWidth = 1;
      octx.setLineDash([3, 3]);
      octx.strokeRect(bx * sx, by * sy, bw * sx, bh * sy);
      octx.setLineDash([]);
      octx.fillStyle = octx.strokeStyle;
      octx.fillText(`${p.class} ${Math.round(p.score * 100)}%`, bx * sx, by * sy - 2);
    }

    // 深度圖迷你預覽
    if (lastDepthMap && useMidas) {
      const ps = 80;
      const px = overlay.width - ps - 8, py = overlay.height - ps - 24;
      let dMin = Infinity, dMax = -Infinity;
      for (let i = 0; i < lastDepthMap.length; i++) {
        if (lastDepthMap[i] < dMin) dMin = lastDepthMap[i];
        if (lastDepthMap[i] > dMax) dMax = lastDepthMap[i];
      }
      const range = dMax - dMin || 1;
      const id = octx.createImageData(ps, ps);
      const ms = MIDAS_INPUT_SIZE;
      for (let y = 0; y < ps; y++) {
        for (let x = 0; x < ps; x++) {
          const mx = Math.floor(x / ps * ms), my = Math.floor(y / ps * ms);
          const val = Math.round((lastDepthMap[my * ms + mx] - dMin) / range * 255);
          const idx = (y * ps + x) * 4;
          id.data[idx] = val; id.data[idx+1] = val * 0.6;
          id.data[idx+2] = 255 - val; id.data[idx+3] = 180;
        }
      }
      octx.putImageData(id, px, py);
      octx.strokeStyle = 'rgba(255,255,255,0.3)';
      octx.strokeRect(px, py, ps, ps);
      octx.fillStyle = 'rgba(255,255,255,0.5)';
      octx.font = '9px monospace';
      octx.fillText('depth', px + 2, py - 2);
    }    octx.fillStyle = 'rgba(255,255,255,0.7)';
    octx.font = '11px monospace';
    const nV = predictions.filter(p => VEHICLE_CLASSES.has(p.classId)).length;
    const nL = predictions.filter(p => p.classId === LIGHT_CLASS).length;
    octx.fillText(`det:${predictions.length} car:${nV} light:${nL} ${vw}×${vh} ${ml}${dl}${ll}`, 8, 16);
  }

  // 光流特徵點（低解析度座標 → overlay）
  if (debugMode && cvReady && (ofFgPts || ofBgPts)) {
    const ofSx = overlay.width / OF_INPUT_W;
    const ofSy = overlay.height / OF_INPUT_H;
    if (ofFgPts) {
      octx.fillStyle = 'rgba(255,80,80,0.9)';
      for (let i = 0; i < ofFgPts.rows; i++) {
        const x = ofFgPts.data32F[i * 2] * ofSx;
        const y = ofFgPts.data32F[i * 2 + 1] * ofSy;
        octx.beginPath(); octx.arc(x, y, 2, 0, Math.PI * 2); octx.fill();
      }
    }
    if (ofBgPts) {
      octx.fillStyle = 'rgba(80,200,255,0.7)';
      for (let i = 0; i < ofBgPts.rows; i++) {
        const x = ofBgPts.data32F[i * 2] * ofSx;
        const y = ofBgPts.data32F[i * 2 + 1] * ofSy;
        octx.beginPath(); octx.arc(x, y, 1.5, 0, Math.PI * 2); octx.fill();
      }
    }
    // FOE
    octx.strokeStyle = 'rgba(255,255,0,0.8)';
    octx.lineWidth = 1;
    const fx = OF_INPUT_W * 0.5 * ofSx;
    const fy = OF_INPUT_H * OF_FOE_Y_RATIO * ofSy;
    octx.beginPath(); octx.moveTo(fx - 8, fy); octx.lineTo(fx + 8, fy);
    octx.moveTo(fx, fy - 8); octx.lineTo(fx, fy + 8); octx.stroke();
  }
}

// =============================================
// 核心偵測迴圈
// =============================================
async function detect() {
  if (!running || screenOff || !video.videoWidth) {
    if (running && !screenOff) scheduleNext();
    return;
  }
  const vw = video.videoWidth, vh = video.videoHeight;

  // 1) 物件偵測
  const predictions = await detectObjects(video);
  // 2) 自車運動（GPS 測速）
  updateEgoMotion();
  // 3) 深度估測（靜止 + 有車輛時才跑，且每 N 幀一次省電）
  if (useMidas && midasSession) {
    const hasVeh = predictions.some(p => VEHICLE_CLASSES.has(p.classId));
    depthFrameCounter++;
    if (!egoMoving && hasVeh && depthFrameCounter >= DEPTH_RUN_EVERY) {
      depthFrameCounter = 0;
      try { lastDepthMap = await midasEstimateDepth(video); }
      catch (e) { lastDepthMap = null; }
    }
  }

  // 3.5) 車道線偵測 (UFLD，每 N 幀一次省電)
  if (useUfld && ufldSession) {
    ufldFrameCounter++;
    if (ufldFrameCounter >= UFLD_RUN_EVERY) {
      ufldFrameCounter = 0;
      try {
        lastLanes = await ufldDetectLanes(video);
        lastEgoLane = findEgoLane(lastLanes, vw, vh);
      } catch (e) {
        console.warn('[A-Eye] UFLD 推論失敗:', e.message);
        lastLanes = null;
        lastEgoLane = null;
      }
    }
  }
  // 4) 分類
  const vehicles = predictions.filter(p => VEHICLE_CLASSES.has(p.classId));
  const lights = predictions.filter(p => p.classId === LIGHT_CLASS);
  // 5) 前車（帶深度）— 可由 UI 關閉
  const carAlerts = enableCarDepart
    ? updateCarTracking(vehicles, vw, vh, lastDepthMap)
    : [];

  // 5.5) 光流起步判定 — 只在 TRACKING 狀態 + 有 smoothedBbox + cv ready 時執行
  if (enableCarDepart && cvReady && sysState === STATE.TRACKING && smoothedBbox) {
    try {
      const ofRes = ofTick(smoothedBbox, vw, vh);
      if (ofRes.departed) {
        const a = tryAlert('move', '🚗 前車已起步！');
        if (a) carAlerts.push(a);
        // 觸發後重置，避免連續警示同一事件
        ofReleaseAll();
      }
      if (debugMode && ofRes.info) {
        carAlerts.push({ type: 'idle', text: `OF: ${ofRes.info}` });
      }
    } catch (e) {
      console.warn('[A-Eye] 光流錯誤:', e.message);
      ofReleaseAll();
    }
  } else if (!smoothedBbox || sysState !== STATE.TRACKING) {
    // 失去追蹤 → 重置光流
    if (ofPrevGray || ofFgPts) ofReleaseAll();
  }

  // 6) 紅綠燈 — 可由 UI 關閉
  const lightAlerts = enableTrafficLight
    ? updateTrafficLight(lights)
    : [];

  // 7) 合併
  const alerts = [...carAlerts, ...lightAlerts];
  if (alerts.length === 0) {
    if (!enableCarDepart && !enableTrafficLight) {
      alerts.push({ type: 'idle', text: '⚠️ 所有偵測項目皆已關閉' });
    } else {
      alerts.push({ type: 'idle', text: '👀 偵測中...' });
    }
  }

  // 8) 繪製
  drawOverlay(predictions, vw, vh);
  renderAlerts(alerts);
  scheduleNext();
}

function scheduleNext() {
  if (running) setTimeout(detect, DETECT_INTERVAL);
}

// =============================================
// UI
// =============================================
function renderAlerts(alerts) {
  alertsEl.innerHTML = alerts.map(a =>
    `<div class="alert-badge ${a.type}">${a.text}</div>`
  ).join('');
}

function resetAllState() {
  resetCarState();
  sysState = STATE.IDLE;  lockedLight = null;  trafficLightState = 'unknown';
  lightConfirmCount = 0;  egoMoving = false;
  lastDepthMap = null;
  lastLanes = null;
  lastEgoLane = null;
  ufldFrameCounter = 0;
  if (typeof ofReleaseAll === 'function') ofReleaseAll();
}

const BTN_ICON = '<img src="icon-192.png" class="btn-icon">';

// =============================================
// 功能開關 UI
// =============================================
function onFeatureToggle() {
  const carCb = document.getElementById('enable-car');
  const lightCb = document.getElementById('enable-light');
  enableCarDepart = !!(carCb && carCb.checked);
  enableTrafficLight = !!(lightCb && lightCb.checked);

  // 視覺樣式
  const carLabel = document.getElementById('toggle-car-label');
  const lightLabel = document.getElementById('toggle-light-label');
  if (carLabel) carLabel.classList.toggle('active', enableCarDepart);
  if (lightLabel) lightLabel.classList.toggle('active', enableTrafficLight);

  // 關閉前車偵測時，清空前車狀態避免殘留
  if (!enableCarDepart) {
    resetCarState();
    sysState = STATE.IDLE;
  }
  // 關閉紅綠燈偵測時，清空紅綠燈狀態
  if (!enableTrafficLight) {
    lockedLight = null;
    trafficLightState = 'unknown';
    lightConfirmCount = 0;
  }

  // 持久化
  try {
    localStorage.setItem('aeye.enableCarDepart', enableCarDepart ? '1' : '0');
    localStorage.setItem('aeye.enableTrafficLight', enableTrafficLight ? '1' : '0');
  } catch (e) { /* ignore */ }
}

// 初始化：從 localStorage 讀回偏好並套用到 UI
(function initFeatureToggles() {
  try {
    const c = localStorage.getItem('aeye.enableCarDepart');
    const l = localStorage.getItem('aeye.enableTrafficLight');
    if (c !== null) enableCarDepart = c === '1';
    if (l !== null) enableTrafficLight = l === '1';
  } catch (e) { /* ignore */ }

  const apply = () => {
    const carCb = document.getElementById('enable-car');
    const lightCb = document.getElementById('enable-light');
    if (carCb) carCb.checked = enableCarDepart;
    if (lightCb) lightCb.checked = enableTrafficLight;
    onFeatureToggle();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply, { once: true });
  } else {
    apply();
  }
})();

async function toggleDetection() {if (running) {
    running = false;
    stopCamera();
    stopGps();
    if (wakeLock) { wakeLock.release(); wakeLock = null; }
    if (document.pictureInPictureElement) document.exitPictureInPicture();
    resetAllState();
    statusDot.className = 'inactive';
    statusText.textContent = '已停止';
    toggleBtn.innerHTML = `${BTN_ICON} 開始偵測`;
    toggleBtn.className = 'start';
    pipBtn.style.display = 'none';
    octx.clearRect(0, 0, overlay.width, overlay.height);
    renderAlerts([{ type: 'idle', text: '⏳ 等待啟動...' }]);
    return;
  }

  try {
    toggleBtn.disabled = true;
    toggleBtn.innerHTML = `${BTN_ICON} 啟動中...`;    await startCamera();
    await loadModels();
    startGps();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    await requestWakeLock();

    running = true;
    statusDot.className = '';
    statusText.textContent = '偵測中';
    toggleBtn.innerHTML = `${BTN_ICON} 停止`;
    toggleBtn.className = 'stop';
    toggleBtn.disabled = false;
    pipBtn.style.display = '';

    scheduleNext();
  } catch (e) {
    toggleBtn.disabled = false;
    toggleBtn.innerHTML = `${BTN_ICON} 開始偵測`;
    toggleBtn.className = 'start';
    statusText.textContent = '❌ ' + (e.message || '啟動失敗');
    renderAlerts([{ type: 'idle', text: '請確認相機權限並重試' }]);
  }
}

// =============================================
// 螢幕關閉
// =============================================
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    screenOff = true;
    screenOffTimer = setTimeout(() => {
      if (running) toggleDetection();
    }, SCREEN_OFF_TIMEOUT);
  } else {
    screenOff = false;
    clearTimeout(screenOffTimer);
    if (running) requestWakeLock();
  }
});

// =============================================
// Service Worker
// =============================================
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
