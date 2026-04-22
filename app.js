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
const DETECT_INTERVAL = 600;

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
const UFLD_RUN_EVERY = 5;        // 每 N 幀跑一次（省電）
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

// 紅綠燈
let lockedLight = null;
let trafficLightState = 'unknown';
let lightConfirmCount = 0;

// 深度圖
let lastDepthMap = null;
let depthFrameCounter = 0;
const DEPTH_RUN_EVERY = 3;  // 每 N 幀才跑一次 MiDaS

// 自車運動
let egoMoving = false;
let gpsWatchId = null;               // GPS watchPosition ID
let lastGpsSpeed = null;             // 最新 GPS 速度 (m/s)，null = 無 GPS
let lastGpsTime = 0;                 // 最後收到 GPS 的時間

// 車道線 (UFLD)
let lastLanes = null;                // 最近一次車道線結果 [{x, y}[], ...]
let lastEgoLane = null;              // {leftX(y), rightX(y)} 自車車道邊界函數
let ufldFrameCounter = 0;

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

// =============================================
// 載入模型
// =============================================
async function loadModels() {
  statusText.textContent = '載入 AI 模型中...';

  // 嘗試載入 YOLOv8n ONNX
  if (typeof ort !== 'undefined') {
    try {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

      statusText.textContent = '載入 YOLOv8n 模型...';
      yoloSession = await ort.InferenceSession.create('./models/yolov8n.onnx', {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });
      useOnnx = true;
      console.log('[A-Eye] YOLOv8n ONNX 載入成功');

      // 嘗試載入 MiDaS
      try {
        statusText.textContent = '載入深度模型...';
        midasSession = await ort.InferenceSession.create('./models/midas_small.onnx', {
          executionProviders: ['webgl', 'wasm'],
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
      }

    } catch (e) {
      console.warn('[A-Eye] YOLOv8n 載入失敗，降級到 COCO-SSD:', e.message);
      useOnnx = false;
    }
  }
  // Fallback: COCO-SSD（動態載入 TF.js + COCO-SSD）
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
  }

  const info = useOnnx ? `YOLOv8n${useMidas ? ' + MiDaS' : ''}${useUfld ? ' + UFLD' : ''}` : 'COCO-SSD';
  statusText.textContent = `模型: ${info}`;
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
  const now = Date.now();

  switch (sysState) {

    case STATE.IDLE: {
      if (vehicles.length === 0) break;
      let best = null, bestS = 0;
      for (const v of vehicles) {
        const s = scoreFrontCar(v.bbox, vw, vh, depthMap);
        if (s > bestS) { bestS = s; best = v; }
      }
      if (best && bestS > (depthMap ? 0.5 : 0.002)) {
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
      }

      if (matched && bestIoU > IOU_LOCK_MATCH) {
        disappearFrameCount = 0;
        smoothedBbox = emaSmooth(smoothedBbox, matched, EMA_ALPHA);        const area = smoothedBbox.w * smoothedBbox.h;
        if (egoMoving) {
          // 行駛中：不做任何警示判定
          moveConfirmCount = 0;
          bboxHistory = [];
          alerts.push({ type: 'idle', text: '🚙 行駛中' });
        } else {
          bboxHistory.push({ area, y: smoothedBbox.y, time: now });
          if (bboxHistory.length > MOVE_HISTORY) bboxHistory.shift();

          if (bboxHistory.length >= MOVE_HISTORY) {
            // 用中位數取代首尾比較，抗抖動
            const areas = bboxHistory.map(h => h.area);
            const ys = bboxHistory.map(h => h.y);
            const halfLen = Math.floor(MOVE_HISTORY / 2);
            const oldAreas = areas.slice(0, halfLen);
            const newAreas = areas.slice(-halfLen);
            const oldYs = ys.slice(0, halfLen);
            const newYs = ys.slice(-halfLen);

            const medianOldArea = median(oldAreas);
            const medianNewArea = median(newAreas);
            const medianOldY = median(oldYs);
            const medianNewY = median(newYs);

            const areaShrink = (medianOldArea - medianNewArea) / (medianOldArea || 1);
            const yRise = (medianOldY - medianNewY) / vh;
            const threshold = moveConfirmCount > 0 ? MOVE_EXIT_RATIO : MOVE_ENTER_RATIO;
            const yThreshold = moveConfirmCount > 0 ? 0.012 : MOVE_Y_ENTER;

            if (areaShrink > threshold || yRise > yThreshold) {
              moveConfirmCount++;
              if (moveConfirmCount >= MOVE_CONFIRM_FRAMES) {
                const a = tryAlert('move', '🚗 前車已起步！');
                if (a) alerts.push(a);
                resetCarState();
                sysState = STATE.IDLE;
                break;
              }
            } else {
              if (moveConfirmCount > 0) moveConfirmCount--;  // 漸減而非歸零，更穩健
            }
          }
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
    }

    octx.fillStyle = 'rgba(255,255,255,0.7)';
    octx.font = '11px monospace';
    const nV = predictions.filter(p => VEHICLE_CLASSES.has(p.classId)).length;
    const nL = predictions.filter(p => p.classId === LIGHT_CLASS).length;
    octx.fillText(`det:${predictions.length} car:${nV} light:${nL} ${vw}×${vh} ${ml}${dl}${ll}`, 8, 16);
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

  // 5) 前車（帶深度）
  const carAlerts = updateCarTracking(vehicles, vw, vh, lastDepthMap);

  // 6) 紅綠燈
  const lightAlerts = updateTrafficLight(lights);

  // 7) 合併
  const alerts = [...carAlerts, ...lightAlerts];
  if (alerts.length === 0) alerts.push({ type: 'idle', text: '👀 偵測中...' });

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
}

const BTN_ICON = '<img src="icon-192.png" class="btn-icon">';

async function toggleDetection() {  if (running) {
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
