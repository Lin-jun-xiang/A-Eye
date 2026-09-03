// =============================================
// 離線回放評測（replay harness）
// =============================================
// 這是我認為最該優先建立的東西：v6 沒有任何客觀指標，
// 所以每次調參都只能憑「今天路上感覺比較少誤報」判斷 —— 這不可能收斂。
//
// 用法：
//   1. 用 App 的 ⏺ 按鈕錄一段（會產出 .webm / .sensors.jsonl / .meta.json）
//   2. 錄影時前車一起步就按 🚩，那些時間點就是 ground truth
//   3. 把三個檔案丟進這個頁面 → 輸出 誤警/小時、漏報率、偵測延遲
//
// 與即時 App 共用完全同一份 pipeline，所以這裡量到的就是真機行為。

import { CONFIG } from '../config.js';
import { Pipeline } from '../core/pipeline.js';
import { Detector } from '../perception/detector.js';
import { ImuSensor } from '../sensors/imu.js';

// 判定命中的時間窗：標註點前 0.5 秒到後 3 秒內的警示算命中
const MATCH_BEFORE_MS = 500;
const MATCH_AFTER_MS = 3000;

/** 從錄下的感測器 log 驅動的 IMU（介面與 ImuSensor 完全相同） */
class ReplayImu extends ImuSensor {
  constructor(cfg, rows) {
    super(cfg);
    this.rows = rows.filter((r) => r.gyro || r.accel);
    this.cursor = 0;
    this.available = this.rows.length > 0;
  }
  /** 把 samples 填到模擬時刻 t 為止 */
  seek(t) {
    while (this.cursor < this.rows.length && this.rows[this.cursor].t <= t) {
      const r = this.rows[this.cursor++];
      const g = r.gyro || [0, 0, 0];
      const a = r.accel || [0, 0, 0];
      this.samples.push({
        t: r.t,
        rx: g[0], ry: g[1], rz: g[2],
        ax: a[0], ay: a[1], az: a[2],
        hasGyro: !!r.gyro,
      });
    }
    const cutoff = t - this.cfg.imu.bufferMs;
    while (this.samples.length && this.samples[0].t < cutoff) this.samples.shift();
  }
  async start() { return this.available; }
  stop() {}
}

/** 從錄下的感測器 log 驅動的 GPS */
class ReplayGps {
  constructor(cfg, rows) {
    this.cfg = cfg;
    this.rows = rows;
    this.cursor = 0;
    this.speed = null;
    this.ts = 0;
  }
  seek(t) {
    while (this.cursor < this.rows.length && this.rows[this.cursor].t <= t) {
      const r = this.rows[this.cursor++];
      if (r.gpsSpeed !== undefined) {
        this.speed = r.gpsSpeed;
        this.ts = r.t;
      }
    }
  }
  currentSpeed(now) {
    if (this.speed === null) return null;
    if (now - this.ts > this.cfg.ego.gpsStaleMs) return null;
    return this.speed;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve;
    s.onerror = () => reject(new Error('load fail ' + src));
    document.head.appendChild(s);
  });
}

function parseJsonl(text) {
  return text.split('\n').map((l) => l.trim()).filter(Boolean)
    .map((l) => { try { return JSON.parse(l); } catch (_) { return null; } })
    .filter(Boolean);
}

export class ReplayRunner {
  constructor({ video, canvas, onProgress, onLog }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.onProgress = onProgress || (() => {});
    this.onLog = onLog || (() => {});
    this.abort = false;
  }

  async load({ videoFile, sensorFile, metaFile }) {
    this.videoUrl = URL.createObjectURL(videoFile);
    this.video.src = this.videoUrl;
    await new Promise((res, rej) => {
      this.video.onloadedmetadata = res;
      this.video.onerror = () => rej(new Error('影片無法解碼'));
    });
    this.sensorRows = sensorFile ? parseJsonl(await sensorFile.text()) : [];
    this.meta = metaFile ? JSON.parse(await metaFile.text()) : null;
    this.marks = (this.meta && this.meta.marks) ? this.meta.marks.filter((m) => m.label === 'depart') : [];
    return {
      duration: this.video.duration,
      vw: this.video.videoWidth,
      vh: this.video.videoHeight,
      sensorRows: this.sensorRows.length,
      marks: this.marks.length,
    };
  }

  async seekTo(tSec) {
    if (Math.abs(this.video.currentTime - tSec) < 1e-4) return;
    return new Promise((res) => {
      const done = () => { this.video.removeEventListener('seeked', done); res(); };
      this.video.addEventListener('seeked', done);
      this.video.currentTime = tSec;
    });
  }

  /**
   * @param opts { hz, cfgOverride }
   */
  async run({ hz = 15, cfgOverride = null } = {}) {
    this.abort = false;
    const cfg = cfgOverride ? mergeDeep(structuredClone(CONFIG), cfgOverride) : CONFIG;

    const gps = new ReplayGps(cfg, this.sensorRows);
    const imu = new ReplayImu(cfg, this.sensorRows);
    const pipeline = new Pipeline({ cfg, gps, imu });
    const detector = new Detector(cfg);

    this.onLog('載入偵測模型...');
    const info = await detector.init();
    this.onLog(`模型 ${info.model} @${info.provider} ${info.inputSize}px`);

    this.onLog('載入 OpenCV.js...');
    const cvOk = await pipeline.initCv(loadScript);
    if (!cvOk) { this.onLog('❌ OpenCV 載入失敗，無法評測起步判定'); return null; }

    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    this.canvas.width = vw;
    this.canvas.height = vh;

    const dur = this.video.duration;
    const dt = 1 / hz;
    const events = [];
    let steps = 0;
    const t0Wall = performance.now();

    for (let t = 0; t < dur && !this.abort; t += dt) {
      await this.seekTo(t);
      this.ctx.drawImage(this.video, 0, 0, vw, vh);

      const nowSim = t * 1000;
      gps.seek(nowSim);
      imu.seek(nowSim);

      // 離線模式：每一步都確實跑偵測（不掉幀，確保決定性）
      const boxes = await detector.runOnce(this.canvas, nowSim);
      pipeline.onDetections(boxes, nowSim);

      const { events: evs } = pipeline.tick({ source: this.canvas, vw, vh, now: nowSim });
      for (const ev of evs) events.push({ t: nowSim, kind: ev.kind, text: ev.text });

      steps++;
      if (steps % 5 === 0) {
        this.onProgress(t / dur, {
          t, steps, events: events.length,
          wallMs: performance.now() - t0Wall,
          debug: pipeline.debugLines(),
        });
      }
    }

    detector.dispose();
    const report = this.score(events, dur, pipeline);
    this.onProgress(1, { t: dur, steps, events: events.length, wallMs: performance.now() - t0Wall });
    return report;
  }

  /** 把事件與 ground truth 標註配對，算出 FP/小時、漏報、延遲 */
  score(events, durationSec, pipeline) {
    const departs = events.filter((e) => e.kind === 'depart');
    const greens = events.filter((e) => e.kind === 'green');
    const marks = this.marks.slice().sort((a, b) => a.t - b.t);

    const usedEvent = new Set();
    const latencies = [];
    let tp = 0;

    for (const m of marks) {
      let bestIdx = -1, bestDelta = Infinity;
      for (let i = 0; i < departs.length; i++) {
        if (usedEvent.has(i)) continue;
        const d = departs[i].t - m.t;
        if (d < -MATCH_BEFORE_MS || d > MATCH_AFTER_MS) continue;
        if (Math.abs(d) < Math.abs(bestDelta)) { bestDelta = d; bestIdx = i; }
      }
      if (bestIdx >= 0) {
        usedEvent.add(bestIdx);
        tp++;
        latencies.push(bestDelta);
      }
    }

    const fp = departs.length - usedEvent.size;
    const fn = marks.length - tp;
    const hours = durationSec / 3600;
    latencies.sort((a, b) => a - b);

    return {
      durationSec,
      marks: marks.length,
      departEvents: departs.length,
      greenEvents: greens.length,
      tp, fp, fn,
      fpPerHour: hours > 0 ? fp / hours : 0,
      recall: marks.length ? tp / marks.length : null,
      latencyMedianMs: latencies.length ? latencies[Math.floor(latencies.length / 2)] : null,
      latencyP90Ms: latencies.length ? latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * 0.9))] : null,
      latencies,
      flowStats: pipeline.stats,
      events,
    };
  }
}

function mergeDeep(base, over) {
  for (const k of Object.keys(over)) {
    if (over[k] && typeof over[k] === 'object' && !Array.isArray(over[k])) {
      base[k] = mergeDeep(base[k] || {}, over[k]);
    } else {
      base[k] = over[k];
    }
  }
  return base;
}
