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
const recorder = new SessionRecorder({ video, gps, imu });

let running = false;
let starting = false;
let debugMode = false;
let wakeLock = null;
let hidden = false;
let hiddenTimer = null;
let vw = 0, vh = 0;
let lastEventText = '';
let lastEventTs = 0;
const eventLog = [];

const BTN_ICON = '<svg class="btn-icon"><use href="#logo"/></svg>';

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
    lastEventTs = now;
    eventLog.push({ t: Math.round(now), kind: ev.kind });
    if (CONFIG.debug.logEvents) {
      console.log(`[A-Eye] ${ev.text}`, pipeline.departure.debugLine());
    }
  }

  // 繪製
  const t2 = performance.now();
  overlay.draw(hud, vw, vh, { debug: debugMode });
  metrics.drawMs.push(performance.now() - t2);

  renderBadges(hud, now);
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

function renderBadges(hud, now) {
  const badges = [];

  // 剛觸發的事件停留 3 秒
  if (lastEventText && now - lastEventTs < 3000) {
    badges.push({ type: lastEventText.includes('綠燈') ? 'green' : 'move', text: lastEventText });
  }

  if (!pipeline.enableCarDepart && !pipeline.enableTrafficLight) {
    badges.push({ type: 'idle', text: '⚠️ 所有偵測項目皆已關閉' });
    alerts.render(badges);
    return;
  }

  if (hud.ego === EgoState.MOVING) {
    badges.push({ type: 'idle', text: `🚙 ${hud.egoLabel} — 靜默中` });
  } else if (hud.ego === EgoState.UNKNOWN) {
    badges.push({ type: 'idle', text: '❓ 自車狀態未知（等待 GPS / IMU）' });
  }

  if (pipeline.enableCarDepart) {
    if (!pipeline.flow.cvReady) {
      badges.push({ type: 'idle', text: '⏳ 載入光流引擎中...' });
    } else if (!hud.target) {
      badges.push({ type: 'idle', text: '👀 尋找前車...' });
    } else {
      const d = hud.departure;
      const ttc = isFinite(d.ttc) && d.ttc < 999 ? ` TTC ${d.ttc.toFixed(0)}s` : '';
      const pct = Math.round(100 * Math.max(
        Math.min(d.z / d.zFire, 1), 0
      ));
      badges.push({
        type: 'idle',
        text: `🚗 追蹤前車 #${hud.target.id}｜證據 ${pct}%${ttc}`
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
async function start() {
  if (starting || running) return;
  starting = true;
  toggleBtn.disabled = true;
  toggleBtn.innerHTML = `${BTN_ICON} 啟動中...`;
  try {
    setStatus('開啟相機...', true);
    const dim = await frames.startCamera(CONFIG.camera);
    vw = dim.vw; vh = dim.vh;

    // IMU 必須在使用者手勢的呼叫堆疊裡要求授權（iOS 限制）
    setStatus('要求動作感測器授權...', true);
    const imuOk = await imu.start();
    if (!imuOk) console.warn('[A-Eye] IMU 不可用:', imu.permission);

    setStatus('載入偵測模型...', true);
    const info = await detector.init();
    detector.onResult = (boxes, ts, timing) => {
      metrics.detectRate.mark(performance.now());
      metrics.onDetectorResult(timing, detector.latencyMs);
      pipeline.onDetections(boxes, ts);
    };

    gps.start();
    alerts.ensureAudio();
    await requestWakeLock();

    running = true;
    starting = false;
    toggleBtn.disabled = false;
    toggleBtn.innerHTML = `${BTN_ICON} 停止`;
    toggleBtn.className = 'stop';
    recBtn.style.display = '';
    pipBtn.style.display = '';
    const camSet = frames.settings();
    setStatus(
      `${info.model} @${info.provider} ${info.inputSize}px`
      + (camSet ? ` · ${camSet.width}x${camSet.height}` : '')
      + (imuOk ? ' · IMU' : ' · 無IMU'),
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
  frames.stopCamera();
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
