// ===== A-Eye v4 — ADAS-Inspired 起步警示 =====
//
// 架構參考自適應巡航 (ACC) / 前車碰撞警示 (FCW)：
//   1. 狀態機（State Machine）驅動，明確狀態轉移
//   2. Bbox EMA 平滑，消除偵測抖動
//   3. 路面光流偵測自車運動（不依賴物件數量）
//   4. 自車道判定（消失點原理）
//   5. 遲滯門檻（Hysteresis）防止邊界震盪

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
const DETECT_INTERVAL = 600;          // ~1.7 FPS

// 狀態機門檻
const LOCK_ENTER_FRAMES = 3;         // 連續 3 幀才鎖定
const DISAPPEAR_EXIT_FRAMES = 5;     // 連續 5 幀消失才解鎖

// 前車移動判定（遲滯）
const MOVE_ENTER_RATIO = 0.05;       // 面積縮小 5% → 進入 moving
const MOVE_EXIT_RATIO = 0.02;        // 回到 2% 以內才退出 moving
const MOVE_Y_ENTER = 0.025;          // Y 上移 2.5% → 進入
const MOVE_HISTORY = 6;              // 歷史幀數（~3.6 秒）
const MOVE_CONFIRM_FRAMES = 2;       // 連續 2 幀超門檻才觸發

// EMA 平滑
const EMA_ALPHA = 0.4;               // 0=全靠歷史, 1=全靠當前

// IoU
const IOU_LOCK_MATCH = 0.2;          // 鎖定匹配門檻
const IOU_CANDIDATE = 0.3;           // 候選匹配門檻

// 自車運動 — 路面光流
const FLOW_CANVAS_W = 120;           // 光流分析解析度
const FLOW_SAMPLE_ROWS = 3;          // 路面取樣行數
const FLOW_MOVE_THRESHOLD = 4;       // 路面位移 px → 自車在動
const FLOW_STILL_THRESHOLD = 2;      // < 2px → 靜止（遲滯）

// 紅綠燈
const LIGHT_CONFIRM_FRAMES = 2;      // 紅→綠需連續確認

// 其他
const ALERT_COOLDOWN = 4000;
const SCREEN_OFF_TIMEOUT = 5 * 60 * 1000;

// =============================================
// 狀態機
// =============================================
// 系統狀態：IDLE → LOCKING → TRACKING → ALERTING
//                                ↑         ↓
//                                ←─────────←
const STATE = {
  IDLE: 'idle',           // 未偵測到前車
  LOCKING: 'locking',     // 發現候選，連續確認中
  TRACKING: 'tracking',   // 已鎖定前車，監控中
  DEPARTING: 'departing', // 前車消失確認中
};

let sysState = STATE.IDLE;

// =============================================
// 追蹤資料
// =============================================
let running = false;
let model = null;
let wakeLock = null;
let lastAlertTime = { move: 0, green: 0 };

// 前車
let smoothedBbox = null;            // EMA 平滑後的 bbox {x,y,w,h}
let lockFrameCount = 0;
let disappearFrameCount = 0;
let moveConfirmCount = 0;           // 連續移動確認計數
let bboxHistory = [];               // {area, y, time}

// 前車 ego-lane 分數快取
let lockCandidate = null;           // 鎖定候選 {x,y,w,h}

// 紅綠燈
let lockedLight = null;
let trafficLightState = 'unknown';  // 'red' | 'green' | 'unknown'
let lightConfirmCount = 0;

// 自車運動（路面光流）
let egoMoving = false;
let prevFlowStrip = null;           // 上一幀路面灰度條帶
const flowCanvas = document.createElement('canvas');
const flowCtx = flowCanvas.getContext('2d', { willReadFrequently: true });

// 螢幕
let screenOff = false;
let screenOffTimer = null;

// 紅綠燈顏色分析重用
const tmpCanvas = document.createElement('canvas');
const tmpCtx = tmpCanvas.getContext('2d');

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
    video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
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
    else if (video.requestPictureInPicture) await video.requestPictureInPicture();
  } catch {}
}

// =============================================
// 載入模型
// =============================================
async function loadModel() {
  if (model) return;
  statusText.textContent = '載入 AI 模型中...';
  model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
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

// EMA 平滑 bbox
function emaSmooth(prev, curr, alpha) {
  if (!prev) return { ...curr };
  return {
    x: prev.x * (1 - alpha) + curr.x * alpha,
    y: prev.y * (1 - alpha) + curr.y * alpha,
    w: prev.w * (1 - alpha) + curr.w * alpha,
    h: prev.h * (1 - alpha) + curr.h * alpha,
  };
}

// =============================================
// 自車運動偵測 — 路面光流
// =============================================
// 取畫面下方 1/4 的中央橫條帶，比對兩幀灰度偏移量。
// 路面紋理只在自車移動時才會大幅位移。
// 不依賴物件偵測結果 → 空曠道路也能判定。
function detectEgoMotion() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;

  flowCanvas.width = FLOW_CANVAS_W;
  const aspect = vh / vw;
  flowCanvas.height = Math.round(FLOW_CANVAS_W * aspect);
  flowCtx.drawImage(video, 0, 0, flowCanvas.width, flowCanvas.height);

  const fw = flowCanvas.width, fh = flowCanvas.height;
  // 取底部 20%~30% 的幾行（路面區域）
  const yStart = Math.floor(fh * 0.75);
  const yEnd = Math.floor(fh * 0.90);
  const stripH = yEnd - yStart;
  const imgData = flowCtx.getImageData(0, yStart, fw, stripH);
  const d = imgData.data;

  // 建立灰度條帶 (每行平均一個灰度值 → 寬度方向的 1D 信號)
  const currStrip = new Float32Array(fw);
  for (let x = 0; x < fw; x++) {
    let sum = 0;
    for (let y = 0; y < stripH; y++) {
      const i = (y * fw + x) * 4;
      sum += d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
    }
    currStrip[x] = sum / stripH;
  }

  if (!prevFlowStrip) {
    prevFlowStrip = currStrip;
    return;
  }

  // 1D 互相關找水平偏移量
  const maxShift = 20;
  let bestShift = 0, bestCorr = -Infinity;
  for (let shift = -maxShift; shift <= maxShift; shift++) {
    let corr = 0, count = 0;
    for (let x = maxShift; x < fw - maxShift; x++) {
      const x2 = x + shift;
      if (x2 >= 0 && x2 < fw) {
        const diff = currStrip[x] - prevFlowStrip[x2];
        corr -= diff * diff; // 負 SSD，越大越相似
        count++;
      }
    }
    if (count > 0) corr /= count;
    if (corr > bestCorr) { bestCorr = corr; bestShift = shift; }
  }

  prevFlowStrip = currStrip;

  // 遲滯判定
  const absDrift = Math.abs(bestShift);
  if (egoMoving) {
    egoMoving = absDrift > FLOW_STILL_THRESHOLD;
  } else {
    egoMoving = absDrift > FLOW_MOVE_THRESHOLD;
  }
}

// =============================================
// Ego-Lane 前車篩選
// =============================================
// ADAS 用消失點判斷「同車道」。在手機上近似：
//   - 消失點約在畫面水平中央
//   - 同車道的車：bbox 中心 X 越接近畫面中心 → 越可能是前車
//   - 加上面積大（近）→ 得分高
function scoreFrontCar(bbox, vw, vh) {
  const [x, y, w, h] = bbox;
  const cx = (x + w / 2) / vw;           // 歸一化中心 X (0~1)
  const area = (w * h) / (vw * vh);       // 歸一化面積

  // 偏離中心的懲罰（高斯形）
  const centerDev = Math.abs(cx - 0.5);
  const lanePenalty = Math.exp(-centerDev * centerDev / 0.04); // σ=0.2

  return area * lanePenalty;
}

// =============================================
// 前車追蹤 — 狀態機
// =============================================
function updateCarTracking(vehicles, vw, vh) {
  const alerts = [];
  const now = Date.now();

  switch (sysState) {

    // ─── IDLE：尋找前車 ───
    case STATE.IDLE: {
      if (vehicles.length === 0) break;

      // 選最高分候選
      let best = null, bestS = 0;
      for (const v of vehicles) {
        const s = scoreFrontCar(v.bbox, vw, vh);
        if (s > bestS) { bestS = s; best = v; }
      }
      if (best && bestS > 0.002) {
        const [x, y, w, h] = best.bbox;
        lockCandidate = { x, y, w, h };
        lockFrameCount = 1;
        sysState = STATE.LOCKING;
        alerts.push({ type: 'idle', text: '🚗 鎖定前車中...' });
      }
      break;
    }

    // ─── LOCKING：連續確認候選 ───
    case STATE.LOCKING: {
      // 找與候選 IoU 最高的
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
          // 鎖定成功
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
        // 候選消失，回到 IDLE
        lockCandidate = null;
        lockFrameCount = 0;
        sysState = STATE.IDLE;
      }
      break;
    }

    // ─── TRACKING：監控前車 ───
    case STATE.TRACKING: {
      // 用 IoU 匹配鎖定車輛
      let matched = null, bestIoU = 0;
      for (const v of vehicles) {
        const [x, y, w, h] = v.bbox;
        const iou = calcIoU(smoothedBbox, { x, y, w, h });
        if (iou > bestIoU) { bestIoU = iou; matched = { x, y, w, h, confidence: v.score }; }
      }

      if (matched && bestIoU > IOU_LOCK_MATCH) {
        disappearFrameCount = 0;

        // EMA 平滑
        smoothedBbox = emaSmooth(smoothedBbox, matched, EMA_ALPHA);
        const area = smoothedBbox.w * smoothedBbox.h;

        if (egoMoving) {
          // 自車行駛中：追蹤但不判斷移動，重置歷史
          bboxHistory = [];
          moveConfirmCount = 0;
          alerts.push({ type: 'idle', text: '🚙 行駛中' });
        } else {
          // 自車靜止：記錄歷史，判斷前車是否起步
          bboxHistory.push({ area, y: smoothedBbox.y, time: now });
          if (bboxHistory.length > MOVE_HISTORY) bboxHistory.shift();

          if (bboxHistory.length >= MOVE_HISTORY) {
            const oldest = bboxHistory[0];
            const areaShrink = (oldest.area - area) / oldest.area;
            const yRise = (oldest.y - smoothedBbox.y) / vh;

            // 遲滯判定
            const threshold = moveConfirmCount > 0 ? MOVE_EXIT_RATIO : MOVE_ENTER_RATIO;
            const yThreshold = moveConfirmCount > 0 ? 0.015 : MOVE_Y_ENTER;

            if (areaShrink > threshold || yRise > yThreshold) {
              moveConfirmCount++;
              if (moveConfirmCount >= MOVE_CONFIRM_FRAMES) {
                const a = tryAlert('move', '🚗 前車已起步！');
                if (a) alerts.push(a);
                // 起步後重置，準備追蹤下一輛或重新鎖定
                resetCarState();
                sysState = STATE.IDLE;
                break;
              }
            } else {
              moveConfirmCount = 0;
            }
          }

          alerts.push({ type: 'idle', text: `🚗 追蹤前車中（${Math.round(matched.confidence * 100)}%）` });
        }
      } else {
        // 沒匹配到 → 進入消失確認
        disappearFrameCount = 1;
        sysState = STATE.DEPARTING;
        alerts.push({ type: 'idle', text: '🚗 確認前車狀態...' });
      }
      break;
    }

    // ─── DEPARTING：前車消失確認 ───
    case STATE.DEPARTING: {
      // 還是嘗試匹配
      let matched = null, bestIoU = 0;
      for (const v of vehicles) {
        const [x, y, w, h] = v.bbox;
        const iou = calcIoU(smoothedBbox, { x, y, w, h });
        if (iou > bestIoU) { bestIoU = iou; matched = { x, y, w, h, confidence: v.score }; }
      }

      if (matched && bestIoU > IOU_LOCK_MATCH) {
        // 重新出現，回到 TRACKING
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
  disappearFrameCount = 0;
  moveConfirmCount = 0;
}

// =============================================
// 紅綠燈追蹤 + 顏色分析
// =============================================
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

  const top = lights.reduce((a, b) => a.score > b.score ? a : b);
  const [x, y, w, h] = top.bbox;
  lockedLight = { x, y, w, h };
  return top.bbox;
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

  // 繪製鎖定的前車 (用平滑後的 bbox)
  if (smoothedBbox && (sysState === STATE.TRACKING || sysState === STATE.DEPARTING)) {
    const { x, y, w, h } = smoothedBbox;
    octx.strokeStyle = '#3b82f6';
    octx.lineWidth = 2.5;
    octx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    octx.fillStyle = '#3b82f6';
    octx.font = 'bold 13px sans-serif';
    octx.fillText('前車', x * sx, y * sy - 5);
  }

  // 繪製鎖定的紅綠燈
  if (lockedLight) {
    const { x, y, w, h } = lockedLight;
    octx.strokeStyle = '#facc15';
    octx.lineWidth = 2;
    octx.setLineDash([6, 4]);
    octx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    octx.setLineDash([]);
    octx.fillStyle = '#facc15';
    octx.font = 'bold 13px sans-serif';
    octx.fillText('🚦', x * sx, y * sy - 5);
  }

  // 狀態指示
  octx.font = '11px sans-serif';
  octx.fillStyle = 'rgba(255,255,255,0.4)';
  const stateLabel = egoMoving ? '🚙 行駛中' : `📡 ${sysState.toUpperCase()}`;
  octx.fillText(stateLabel, 8, overlay.height - 8);
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

  // 1) 自車運動偵測（獨立於物件偵測）
  detectEgoMotion();

  // 2) 物件偵測
  const predictions = await model.detect(video, 20, 0.25);

  const vehicles = predictions.filter(p =>
    ['car', 'truck', 'bus', 'motorcycle'].includes(p.class)
  );
  const lights = predictions.filter(p => p.class === 'traffic light');

  // 3) 前車狀態機更新
  const carAlerts = updateCarTracking(vehicles, vw, vh);

  // 4) 紅綠燈更新
  const lightAlerts = updateTrafficLight(lights);

  // 5) 合併警報
  const alerts = [...carAlerts, ...lightAlerts];
  if (alerts.length === 0) alerts.push({ type: 'idle', text: '👀 偵測中...' });

  // 6) 繪製
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
  sysState = STATE.IDLE;
  lockedLight = null;
  trafficLightState = 'unknown';
  lightConfirmCount = 0;
  prevFlowStrip = null;
  egoMoving = false;
}

async function toggleDetection() {
  if (running) {
    running = false;
    stopCamera();
    if (wakeLock) { wakeLock.release(); wakeLock = null; }
    if (document.pictureInPictureElement) document.exitPictureInPicture();
    resetAllState();
    statusDot.className = 'inactive';
    statusText.textContent = '已停止';
    toggleBtn.textContent = '▶ 開始偵測';
    toggleBtn.className = 'start';
    pipBtn.style.display = 'none';
    octx.clearRect(0, 0, overlay.width, overlay.height);
    renderAlerts([{ type: 'idle', text: '⏳ 等待啟動...' }]);
    return;
  }

  try {
    toggleBtn.disabled = true;
    toggleBtn.textContent = '⏳ 啟動中...';
    await startCamera();
    await loadModel();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    await requestWakeLock();

    running = true;
    statusDot.className = '';
    statusText.textContent = '偵測中';
    toggleBtn.textContent = '⏹ 停止';
    toggleBtn.className = 'stop';
    toggleBtn.disabled = false;
    pipBtn.style.display = '';

    scheduleNext();
  } catch (e) {
    toggleBtn.disabled = false;
    toggleBtn.textContent = '▶ 開始偵測';
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
