// =============================================
// 快速影片分析（上傳一支影片 → 輸出「第幾秒偵測到什麼」）
// =============================================
// 與另外兩條路的分工：
//
//   📁 影片模式（index.html）   即時路徑，用來「看」判定對不對
//   analyze.html（這裡）        跑完整支影片，用來「列出事件發生在第幾秒」
//   replay.html                 逐格 seek + ground truth 標註，算誤警/漏報/延遲
//
// 為什麼不用 replay.html 的逐格 seek：
//   seek 的延遲（每次數十毫秒）會主導總時間，而且它的目的是「決定性」——
//   要跑完 189 秒的影片得等好幾分鐘。這裡改成讓影片**播放**、用
//   requestVideoFrameCallback 抓每個解碼出來的幀，處理不完的就跳過。
//
// 關鍵：所有時間都用 `mediaTime`（影片時間）當時鐘餵給 pipeline，
// 不是牆上時間。這樣 dwell、基線、冷卻等時間邏輯全部落在影片時間軸上，
// 播放倍速就只影響「取樣密度」，不影響判定門檻。
//
// 倍速的代價要講清楚：每一幀的處理成本（光流 + 推論）是固定的，
// 所以倍速調高只會讓「每一影片秒處理到的幀數」變少，不會讓判定變快。
// 面板會把實際達到的取樣率印出來，讓這個代價可見。

import { CONFIG } from '../config.js';
import { Pipeline } from '../core/pipeline.js';
import { Detector } from '../perception/detector.js';

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('load fail ' + src));
    document.head.appendChild(s);
  });
}

export class QuickAnalyzer {
  constructor({ video, canvas, onProgress, onLog }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.onProgress = onProgress || (() => {});
    this.onLog = onLog || (() => {});
    this.abort = false;
    this.url = null;
  }

  async load(file) {
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(file);
    this.video.src = this.url;
    this.video.muted = true;
    await new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error('影片載入逾時')), 20000);
      this.video.onloadedmetadata = () => { clearTimeout(t); res(); };
      this.video.onerror = () => { clearTimeout(t); rej(new Error('此影片無法解碼')); };
    });
    return {
      duration: this.video.duration,
      vw: this.video.videoWidth,
      vh: this.video.videoHeight,
    };
  }

  stop() { this.abort = true; }

  /**
   * @param opts.playbackRate 播放倍速（越快 → 每影片秒取樣到的幀數越少）
   * @param opts.from,to      只分析這段區間（秒）
   * @param opts.assumeStill  強制假設自車靜止（影片沒有 GPS/IMU 時的替代方案）
   * @returns { events, stats }
   */
  async run({ playbackRate = 2, from = 0, to = Infinity, assumeStill = false } = {}) {
    this.abort = false;
    const video = this.video;
    const vw = video.videoWidth, vh = video.videoHeight;
    const end = Math.min(to, video.duration);

    if (typeof video.requestVideoFrameCallback !== 'function') {
      throw new Error('此瀏覽器不支援 requestVideoFrameCallback，無法逐幀分析');
    }

    this.canvas.width = vw;
    this.canvas.height = vh;

    const pipeline = new Pipeline({ cfg: CONFIG, gps: null, imu: null });
    pipeline.assumeStill = assumeStill;
    const detector = new Detector(CONFIG);

    this.onLog('載入偵測模型...');
    const info = await detector.init();
    this.onLog(`模型 ${info.model} @${info.provider} ${info.inputSize}px`);

    this.onLog('載入 OpenCV.js（約 10MB，第一次會久一點）...');
    const cvOk = await pipeline.initCv(loadScript);
    this.onLog(cvOk ? 'OpenCV 就緒' : '⚠️ OpenCV 載入失敗 → 只會有剎車燈事件，沒有尺度變化率判定');

    const events = [];
    let frames = 0, detects = 0, lastDetectTs = -Infinity;
    const wall0 = performance.now();

    video.playbackRate = playbackRate;
    if (Math.abs(video.currentTime - from) > 0.05) {
      await new Promise((res) => {
        const done = () => { video.removeEventListener('seeked', done); res(); };
        video.addEventListener('seeked', done);
        video.currentTime = from;
      });
    }
    await video.play();

    await new Promise((resolve) => {
      let done = false;
      let lastCb = performance.now();
      const finish = () => {
        if (done) return;
        done = true;
        clearInterval(watch);
        video.removeEventListener('ended', finish);
        resolve();
      };
      // 看門狗：影片停止推進時（緩衝、解碼失敗、分頁被切到背景）
      // requestVideoFrameCallback 不會再觸發 —— 沒有這道保護，
      // 整個分析會永遠掛在這裡等一個不會來的 callback。
      const watch = setInterval(() => {
        if (performance.now() - lastCb > 8000) {
          this.onLog('⚠️ 影片停止推進（緩衝或分頁在背景），提前結束');
          finish();
        }
      }, 1000);
      video.addEventListener('ended', finish);

      const step = async (_now, meta) => {
        lastCb = performance.now();
        if (this.abort || done) return finish();
        const vt = meta ? meta.mediaTime : video.currentTime;
        if (vt >= end || video.ended) return finish();

        // 影片時間當時鐘 —— 所有時間邏輯（dwell、基線、冷卻）都落在影片時間軸上
        const nowSim = vt * 1000;
        this.ctx.drawImage(video, 0, 0, vw, vh);

        // 偵測依 detectHz 在**影片時間**上節流；await 的期間影片會繼續播，
        // 那些幀就直接跳過（用 mediaTime 當時鐘，跳幀不會讓時間軸錯亂）
        if (nowSim - lastDetectTs >= 1000 / CONFIG.loop.detectHz) {
          lastDetectTs = nowSim;
          try {
            const boxes = await detector.runOnce(this.canvas, nowSim);
            pipeline.onDetections(boxes, nowSim);
            detects++;
          } catch (e) { /* 單幀失敗不該中斷整支影片 */ }
        }

        const { events: evs } = pipeline.tick({ source: this.canvas, vw, vh, now: nowSim });
        for (const ev of evs) events.push({ t: vt, kind: ev.kind, text: ev.text });
        frames++;

        if (frames % 10 === 0) {
          const wall = (performance.now() - wall0) / 1000;
          this.onProgress((vt - from) / Math.max(end - from, 1e-6), {
            vt, frames, detects, wall, events: events.length,
            debug: pipeline.debugLines(),
          });
        }
        if (!done) video.requestVideoFrameCallback(step);
      };
      video.requestVideoFrameCallback(step);
    });

    video.pause();
    detector.dispose();

    const wall = (performance.now() - wall0) / 1000;
    const span = Math.max(Math.min(video.currentTime, end) - from, 1e-6);
    const st = pipeline.stats;
    const failTotal = Object.values(st.flowFail || {}).reduce((a, b) => a + b, 0);
    const attempted = st.flowOk + (st.flowFail?.['fg-fit-fail'] || 0)
      + (st.flowFail?.['bg-fit-fail'] || 0) + (st.flowFail?.['bg-inconsistent'] || 0)
      + (st.flowFail?.['fg-too-few'] || 0) + (st.flowFail?.['bg-too-few'] || 0);

    return {
      events,
      stats: {
        from, to: Math.min(video.currentTime, end), span,
        frames, detects, wall,
        sampleFps: frames / span,             // 每「影片秒」處理到幾幀
        detectFps: detects / span,
        flowOk: st.flowOk,
        flowAttempted: attempted,             // 完成基線的量測次數（不含累積中）
        flowFail: st.flowFail,
        flowTotal: st.flowOk + failTotal,
        cvOk,
        model: `${info.model} @${info.provider} ${info.inputSize}px`,
      },
    };
  }
}

/** 把結果排成人看的文字 —— 這個頁面的產出就是這段文字 */
export function formatReport(rep) {
  const s = rep.stats;
  const lines = [];
  lines.push(`分析區間 ${s.from.toFixed(1)}s ~ ${s.to.toFixed(1)}s（${s.span.toFixed(1)}s）`);
  lines.push(`耗時 ${s.wall.toFixed(0)}s，處理 ${s.frames} 幀`
    + `（每影片秒 ${s.sampleFps.toFixed(1)} 幀、偵測 ${s.detectFps.toFixed(1)} 次）`);
  lines.push(`模型 ${s.model}`);
  lines.push('');
  if (!rep.events.length) {
    lines.push('沒有偵測到任何事件。');
  } else {
    lines.push(`偵測到 ${rep.events.length} 個事件：`);
    for (const e of rep.events) {
      lines.push(`  ${e.t.toFixed(2)}s   ${e.text}`);
    }
  }
  lines.push('');
  lines.push('── 診斷 ──');
  if (!s.cvOk) {
    lines.push('OpenCV 未載入 → 沒有尺度變化率判定（只有剎車燈快路徑）');
  } else if (s.flowAttempted === 0) {
    lines.push('光流一次也沒有完成基線量測 → 檢查是否全程都沒有鎖定前車');
  } else {
    const rate = (s.flowOk / s.flowAttempted * 100).toFixed(0);
    lines.push(`光流 ok ${s.flowOk}/${s.flowAttempted} 次完成基線的量測（${rate}%）`);
    const fails = Object.entries(s.flowFail || {})
      .filter(([k]) => k !== 'accumulating' && k !== 'reanchor')
      .sort((a, b) => b[1] - a[1]);
    if (fails.length) lines.push(`  失敗原因：${fails.map(([k, v]) => `${k}:${v}`).join('  ')}`);
  }
  lines.push(`偵測推論 ${s.detects} 次`);
  return lines.join('\n');
}
