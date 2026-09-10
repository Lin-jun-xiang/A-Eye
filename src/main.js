// =============================================
// A-Eye v7 — 主程式（協調層）
// =============================================
// 這一層只做「連接」：DOM、感測器、偵測器、pipeline、UI。
// 所有判定邏輯都在 src/core/pipeline.js 及其下游模組，且不接觸 DOM，
// 因此可以被 replay.html 用錄好的影片完全一樣地驅動。

import { CONFIG } from './config.js';
import { Pipeline, EgoState } from './core/pipeline.js';
import { Detector } from './perception/detector.js';
import { GpsSensor } from './sensors/gps.js';
import { ImuSensor } from './sensors/imu.js';
import { FrameSource } from './capture/frameSource.js';
import { SessionRecorder } from './capture/recorder.js';
import { AlertPresenter } from './ui/alerts.js';
import { Overlay } from './ui/overlay.js';
import { Metrics, DebugPanel } from './ui/hud.js';
import { AnalysisPanel } from './ui/analysisPanel.js';


// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $('camera');
const statusDot = $('status-dot');
const statusText = $('status-text');
const speedEl = $('speed-display');
const toggleBtn = $('toggle-btn');
const recBtn = $('rec-btn');
const markBtn = $('mark-btn');
const debugBtn = $('debug-btn');
const pipBtn = $('pip-btn');
const fileBtn = $('file-btn');
const fileInput = $('file-input');

// ---------- 元件 ----------
const gps = new GpsSensor(CONFIG);
const imu = new ImuSensor(CONFIG);
const pipeline = new Pipeline({ cfg: CONFIG, gps, imu });
const detector = new Detector(CONFIG);
const frames = new FrameSource(video);
const alerts = new AlertPresenter({ alertsEl: $('alerts'), flashEl: $('flash-overlay') });
const overlay = new Overlay($('overlay'), video);
const metrics = new Metrics();
const debugPanel = new DebugPanel($('debug-panel'));
const analysis = new AnalysisPanel($('analysis-panel'));
const recorder = new SessionRecorder({ video, gps, imu });

let running = false;
let starting = false;
// 影片模式：用檔案取代相機。實車路測一趟只能驗一次，而同一段影片可以在
// 每次改動後重跑 —— 這是把「盲調」變成「可重現」的關鍵。
let fileMode = false;
let debugMode = false;
let wakeLock = null;
let hidden = false;
let hiddenTimer = null;
let vw = 0, vh = 0;
let lastEventText = '';
let lastEventKind = '';
let lastEventTs = 0;
// 徽章穩定化：原本每一幀（30Hz）重建 innerHTML，加上數值本身在跳，
// 結果就是整片閃爍到讀不出來。三道處理：限流、內容沒變就不寫 DOM、
// 以及「剎車燈狀態」要維持最短顯示時間。
let lastBadgeHtml = '';
let lastBadgeTs = 0;
let shownBrake = { state: 'unknown', since: 0, pending: null, pendingSince: 0 };
const eventLog = [];

const BTN_ICON = '<svg class="btn-icon"><use href="#logo"/></svg>';
// lit = 有一對紅燈亮著，但還分不出是尾燈還是剎車燈（只看過一個位準）
const BRAKE_LABEL = { on: '踩著', lit: '亮(未確認)', off: '熄', unknown: '?' };

// ---------- 小工具 ----------
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('script load failed: ' + src));
    document.head.appendChild(s);
  });
}

function setStatus(text, active = running) {
  statusText.textContent = text;
  statusDot.className = active ? '' : 'inactive';
}

// ---------- 功能開關 ----------
function readToggles() {
  const car = $('enable-car').checked;
  const light = $('enable-light').checked;
  pipeline.setFeatures({ car, light });
  $('toggle-car-label').classList.toggle('active', car);
  $('toggle-light-label').classList.toggle('active', light);
  try {
    localStorage.setItem('aeye.car', car ? '1' : '0');
    localStorage.setItem('aeye.light', light ? '1' : '0');
  } catch (_) { /* 隱私模式下會丟例外，忽略即可 */ }
}

function initToggles() {
  try {
    const c = localStorage.getItem('aeye.car');
    const l = localStorage.getItem('aeye.light');
    if (c !== null) $('enable-car').checked = c === '1';
    if (l !== null) $('enable-light').checked = l === '1';
  } catch (_) { /* ignore */ }
  $('enable-car').addEventListener('change', readToggles);
  $('enable-light').addEventListener('change', readToggles);
  // 影片模式的「強制假設靜止」可以在播放中即時切換 —— 比重新載入影片快得多
  $('assume-still').addEventListener('change', () => {
    if (fileMode) pipeline.assumeStill = $('assume-still').checked;
  });
  readToggles();
}

// ---------- 幀迴圈 ----------
function onFrame(now) {
  metrics.frameRate.mark(now);
  if (!video.videoWidth) return;
  vw = video.videoWidth;
  vh = video.videoHeight;

  // 送一幀去 Worker（不 await：主執行緒不能被推論綁住）
  if (detector.shouldSubmit(now)) void detector.submit(video, now);

  const t0 = performance.now();
  const { events, hud } = pipeline.tick({ source: video, vw, vh, now });
  const t1 = performance.now();
  metrics.tickRate.mark(now);
  metrics.tickMs.push(t1 - t0);
  if (hud.flow) {
    metrics.flowRate.mark(now);
    metrics.flowMs.push(pipeline.lastFlowMs);
  }

  // 事件
  for (const ev of events) {
    alerts.fire(ev);
    lastEventText = ev.text;
    lastEventKind = ev.kind;
    lastEventTs = now;
    eventLog.push({ t: Math.round(now), kind: ev.kind, vt: video.currentTime });
    // 影片模式：事件累積成時間軸。「有沒有在正確的時間點觸發」這個問題，
    // 一閃即逝的徽章答不了。
    if (fileMode) analysis.addEvent(video.currentTime || 0, ev.kind, ev.text);
    if (CONFIG.debug.logEvents) {
      console.log(`[A-Eye] ${ev.text}`, pipeline.departure.debugLine());
    }
  }

  // 繪製
  const t2 = performance.now();
  overlay.draw(hud, vw, vh, { debug: debugMode });
  metrics.drawMs.push(performance.now() - t2);

  renderBadges(hud, now);
  if (fileMode) {
    analysis.render(now, hud, video.currentTime || 0, video.duration || 0, detector);
  }
  debugPanel.render(now, metrics, pipeline, detector, [
    `cam ${vw}x${vh}`,
    `events ${eventLog.length}`,
  ]);

  if (speedEl) {
    if (hud.speed !== null) {
      speedEl.style.display = '';
      speedEl.textContent = `${Math.round(hud.speed * 3.6)} km/h`;
    } else {
      speedEl.style.display = '';
      speedEl.textContent = hud.ego === EgoState.UNKNOWN ? '-- km/h' : hud.egoLabel;
    }
  }
}

/**
 * 剎車燈狀態的顯示用穩定化：底層狀態在對比邊界上會短暫跳成 unknown，
 * 直接顯示就會閃。要求新狀態持續 400ms 才換 —— 這只影響「顯示」，
 * 判定本身不受影響（判定有自己的 offConfirmMs）。
 */
function stableBrake(state, now) {
  if (state === shownBrake.state) { shownBrake.pending = null; return state; }
  if (shownBrake.pending !== state) { shownBrake.pending = state; shownBrake.pendingSince = now; }
  if (now - shownBrake.pendingSince >= 400) {
    shownBrake.state = state;
    shownBrake.since = now;
    shownBrake.pending = null;
  }
  return shownBrake.state;
}

function renderBadges(hud, now) {
  // 限流：徽章是給「掃一眼」用的，30Hz 更新沒有任何意義，只會閃
  if (now - lastBadgeTs < 200) return;
  lastBadgeTs = now;
  const badges = [];

  // 剛觸發的事件停留 3 秒（「鬆剎車」只留 1.5 秒，它只是預告）
  const hold = lastEventKind === 'release' ? 1500
    : lastEventKind === 'ready' ? 2500 : 3000;
  if (lastEventText && now - lastEventTs < hold) {
    const type = lastEventKind === 'green' ? 'green'
      : (lastEventKind === 'release' || lastEventKind === 'ready') ? 'red' : 'move';
    badges.push({ type, text: lastEventText });
  }

  if (!pipeline.enableCarDepart && !pipeline.enableTrafficLight) {
    badges.push({ type: 'idle', text: '⚠️ 所有偵測項目皆已關閉' });
    alerts.render(badges);
    return;
  }

  if (fileMode) {
    badges.push({ type: 'idle',
      text: '📁 影片模式' + (hud.assumeStill ? '（強制假設靜止）' : '') });
  }
  if (!hud.assumeStill) {
    if (hud.ego === EgoState.MOVING) {
      badges.push({ type: 'idle', text: `🚙 ${hud.egoLabel} — 靜默中` });
    } else if (hud.ego === EgoState.UNKNOWN) {
      badges.push({ type: 'idle', text: '❓ 自車狀態未知（等待 GPS / IMU）' });
    }
  }

  // 影片模式：詳細資訊都在分析面板裡，底部只留「事件」與「模式」兩條。
  // 兩邊都顯示只會互相重疊，而且同一份資訊出現兩次反而更難讀。
  if (fileMode) {
    if (badges.length === 0) badges.push({ type: 'idle', text: '📁 分析中...' });
    alerts.render(badges);
    return;
  }

  if (pipeline.enableCarDepart) {
    // 光流還沒載好不代表什麼都不能做 —— 剎車燈快路徑不需要 OpenCV，
    // 所以這條只是資訊，不再取代下面的追蹤徽章
    if (!pipeline.flow.cvReady) {
      badges.push({ type: 'idle', text: '⏳ 載入光流引擎中（剎車燈偵測已可用）' });
    }
    if (!hud.target) {
      badges.push({ type: 'idle', text: '👀 尋找前車...' });
    } else {
      const d = hud.departure;
      const ttc = isFinite(d.ttc) && d.ttc < 999 ? ` TTC ${d.ttc.toFixed(0)}s` : '';
      // 量化到 5% 一格：逐幀的小數變動對「還差多少」的判讀沒有幫助，只會閃
      const pct = Math.round(20 * Math.max(Math.min(d.z / d.zFire, 1), 0)) * 5;
      badges.push({
        type: 'idle',
        // 把「為什麼還沒報」直接寫在畫面上 —— 實車測試時手機沒有 console，
        // 只看得到證據百分比的話，完全無法區分「訊號不足」和「某道閘門卡住」。
        text: `🚗 追蹤前車 #${hud.target.id}｜證據 ${pct}%${ttc}｜${d.reason}`
          + `｜剎車燈 ${BRAKE_LABEL[stableBrake(hud.brakeState, now)] || '?'}`
          + (hud.brakeChmslUsable ? '(第三燈)' : '')
          + (hud.brakePrimed ? '⚡已預備' : '')
          + (hud.trusted ? '' : '｜⚠背景不可信'),
      });
    }
  }

  if (pipeline.enableTrafficLight && hud.lightColor && hud.lightColor !== 'unknown') {
    const m = { red: ['red', '🔴 紅燈'], green: ['green', '🟢 綠燈'], yellow: ['red', '🟡 黃燈'] };
    const [type, text] = m[hud.lightColor] || ['idle', ''];
    if (text) badges.push({ type, text });
  }

  if (recorder.recording) {
    badges.push({ type: 'red', text: `⏺ 錄影中（標註 ${recorder.marks.length} 筆）` });
  }

  if (badges.length === 0) badges.push({ type: 'idle', text: '👀 偵測中...' });
  alerts.render(badges);
}

// ---------- 啟停 ----------
async function start(videoFile = null) {
  if (starting || running) return;
  starting = true;
  fileMode = !!videoFile;
  toggleBtn.disabled = true;
  toggleBtn.innerHTML = `${BTN_ICON} 啟動中...`;
  try {
    let imuOk = false;
    if (fileMode) {
      setStatus('載入影片...', true);
      const dim = await frames.startFile(videoFile);
      vw = dim.vw; vh = dim.vh;
      // 影片檔沒有 GPS / IMU，自車是否靜止無從得知。
      // 若照原本的規則（unknown → 靜默），影片模式會一個警示都不出 ——
      // 所以這裡明確假設自車靜止，並在畫面上標示出來，不讓它變成隱性行為。
      // 不再預設假設靜止。影片自己就能判斷自車動不動 ——
      // GPS/IMU 都不可用時，判定會降級到「背景尺度變化率」那一路：
      // 自車前進 → 背景逼近 → 尺度 > 1。用假設取代量測的代價是：
      // 行駛中的影片會被當成停著，於是前車自然的遠離全部變成「前車已起步」。
      pipeline.assumeStill = $('assume-still').checked;
      $('toggle-still-label').style.display = '';
      // 顯示方式與 overlay 的座標映射必須一起切換
      video.classList.add('contain');
      overlay.fit = 'contain';
    } else {
      setStatus('開啟相機...', true);
      const dim = await frames.startCamera(CONFIG.camera);
      vw = dim.vw; vh = dim.vh;
      pipeline.assumeStill = false;
      $('toggle-still-label').style.display = 'none';
      video.classList.remove('contain');
      overlay.fit = 'cover';

      // IMU 必須在使用者手勢的呼叫堆疊裡要求授權（iOS 限制）
      setStatus('要求動作感測器授權...', true);
      imuOk = await imu.start();
      if (!imuOk) console.warn('[A-Eye] IMU 不可用:', imu.permission);
    }

    setStatus('載入偵測模型...', true);
    const info = await detector.init();
    detector.onResult = (boxes, ts, timing) => {
      metrics.detectRate.mark(performance.now());
      metrics.onDetectorResult(timing, detector.latencyMs);
      pipeline.onDetections(boxes, ts);
    };

    if (!fileMode) gps.start();
    alerts.ensureAudio();
    await requestWakeLock();

    running = true;
    starting = false;
    toggleBtn.disabled = false;
    toggleBtn.innerHTML = `${BTN_ICON} 停止`;
    toggleBtn.className = 'stop';
    recBtn.style.display = fileMode ? 'none' : '';
    pipBtn.style.display = '';
    analysis.setVisible(fileMode);
    const camSet = frames.settings();
    setStatus(
      `${info.model} @${info.provider} ${info.inputSize}px`
      + (fileMode ? ` · 影片 ${vw}x${vh}` : '')
      + (camSet ? ` · ${camSet.width}x${camSet.height}` : '')
      + (fileMode ? '' : (imuOk ? ' · IMU' : ' · 無IMU')),
      true
    );

    frames.start(onFrame);

    // OpenCV 體積大（~10MB），非阻斷式載入；載好之前只是還不做起步判定
    pipeline.initCv(loadScript).then((ok) => {
      if (!ok) setStatus('⚠️ 光流引擎載入失敗，起步偵測停用', true);
      else console.log('[A-Eye] OpenCV.js ready');
    });
  } catch (e) {
    starting = false;
    running = false;
    // 影片載入失敗時要收乾淨，否則 blob URL 會留著、下一次啟動狀態也不對
    if (fileMode) { frames.stopFile(); fileMode = false; }
    pipeline.assumeStill = false;
    video.classList.remove('contain');
    overlay.fit = 'cover';
    analysis.setVisible(false);
    $('toggle-still-label').style.display = 'none';
    toggleBtn.disabled = false;
    toggleBtn.innerHTML = `${BTN_ICON} 開始偵測`;
    toggleBtn.className = 'start';
    // 手機上沒有 console，所以錯誤全文要直接顯示在畫面上 ——
    // 只顯示「請確認權限」會把真正的原因（模型載入、CORS、WebGPU…）蓋掉
    const msg = String(e.message || '啟動失敗');
    setStatus('❌ ' + msg.split('\n')[0], false);
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    alerts.render([{
      type: 'red',
      text: `<div style="font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all">${esc(msg)}</div>`,
    }]);
    console.error('[A-Eye] 啟動失敗:', e);
  }
}

async function stop() {
  running = false;
  frames.stop();
  if (recorder.recording) await recorder.stop();
  if (fileMode) frames.stopFile(); else frames.stopCamera();
  fileMode = false;
  pipeline.assumeStill = false;
  video.classList.remove('contain');
  overlay.fit = 'cover';
  analysis.setVisible(false);
  $('toggle-still-label').style.display = 'none';
  gps.stop();
  imu.stop();
  detector.dispose();
  pipeline.reset();
  if (wakeLock) { try { wakeLock.release(); } catch (_) {} wakeLock = null; }
  if (document.pictureInPictureElement) { try { await document.exitPictureInPicture(); } catch (_) {} }
  overlay.clear();
  toggleBtn.innerHTML = `${BTN_ICON} 開始偵測`;
  toggleBtn.className = 'start';
  recBtn.style.display = 'none';
  markBtn.style.display = 'none';
  pipBtn.style.display = 'none';
  setStatus('已停止', false);
  alerts.render([{ type: 'idle', text: '⏳ 等待啟動...' }]);
}

async function requestWakeLock() {
  try {
    if (navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    }
  } catch (_) { /* 非 HTTPS 或不支援 */ }
}

// ---------- 按鈕 ----------
toggleBtn.addEventListener('click', () => { running ? stop() : start(); });

fileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const f = fileInput.files && fileInput.files[0];
  fileInput.value = '';                  // 允許重選同一個檔案
  if (!f) return;
  if (running) await stop();
  await start(f);
});

debugBtn.addEventListener('click', () => {
  debugMode = !debugMode;
  debugBtn.style.opacity = debugMode ? 1 : 0.4;
  debugPanel.setVisible(debugMode);
});

pipBtn.addEventListener('click', async () => {
  try {
    if (document.pictureInPictureElement) await document.exitPictureInPicture();
    else if (video.requestPictureInPicture) await video.requestPictureInPicture();
  } catch (_) { /* ignore */ }
});

recBtn.addEventListener('click', async () => {
  if (recorder.recording) {
    recBtn.textContent = '⏳';
    const r = await recorder.stop();
    recBtn.textContent = '⏺';
    markBtn.style.display = 'none';
    if (r) {
      alerts.render([{ type: 'idle', text: `✅ 已存檔（${(r.sizeBytes / 1e6).toFixed(1)}MB、${r.marks} 筆標註）` }]);
    }
  } else {
    if (recorder.start()) {
      recBtn.textContent = '⏹';
      markBtn.style.display = '';
    } else {
      alerts.render([{ type: 'idle', text: '❌ 此瀏覽器不支援錄影' }]);
    }
  }
});

markBtn.addEventListener('click', () => {
  const m = recorder.mark('depart');
  if (m) {
    alerts.vibrate('green');
    console.log('[A-Eye] 標註 ground truth @', m.t, 'ms');
  }
});

// ---------- 背景 / 前景 ----------
document.addEventListener('visibilitychange', () => {
  hidden = document.hidden;
  if (hidden) {
    hiddenTimer = setTimeout(() => { if (running) stop(); }, CONFIG.alert.screenOffTimeoutMs);
  } else {
    clearTimeout(hiddenTimer);
    if (running) requestWakeLock();
  }
});

// ---------- 啟動 ----------
initToggles();
debugPanel.setVisible(false);
setStatus('點擊下方按鈕開始偵測', false);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}

// ?debug=1 直接開啟除錯面板
try {
  if (new URLSearchParams(location.search).get('debug') === '1') {
    debugMode = true;
    debugBtn.style.opacity = 1;
    debugPanel.setVisible(true);
  }
} catch (_) { /* ignore */ }

// 給 console 用的檢查窗口（手機上配合 Eruda 很有用）
window.AEye = { CONFIG, pipeline, detector, metrics, gps, imu, recorder, eventLog };
