// =============================================
// 偵測器 facade（主執行緒側）
// =============================================
// 政策：同時只有一個 frame 在飛。忙碌時「丟幀」而不是「排隊」——
// 排隊只會讓結果越來越舊，對即時警示毫無價值。

import { CONFIG } from '../config.js';

export class Detector {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.worker = null;
    this.ready = false;
    this.info = null;
    this.error = null;
    this.inFlight = false;
    this.lastSubmitTs = 0;
    this.lastResultTs = 0;
    this.timing = null;
    this.onResult = null;      // (boxes, ts, timing) => void
    this.latencyMs = 0;
  }

  async init() {
    return new Promise((resolve, reject) => {
      let settled = false;
      try {
        this.worker = new Worker(new URL('./detector.worker.js', import.meta.url));
      } catch (e) {
        this.error = 'Worker 建立失敗: ' + e.message;
        reject(new Error(this.error));
        return;
      }

      this.worker.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === 'ready') {
          this.ready = true;
          this.info = m;
          if (!settled) { settled = true; resolve(m); }
        } else if (m.type === 'error') {
          this.error = m.message;
          if (!settled) { settled = true; reject(new Error(m.message)); }
        } else if (m.type === 'result') {
          this.inFlight = false;
          this.lastResultTs = m.ts;
          this.latencyMs = performance.now() - m.ts;
          if (m.timing) this.timing = m.timing;
          if (m.error) console.warn('[A-Eye] 偵測錯誤:', m.error);
          if (this.onResult) this.onResult(m.boxes || [], m.ts, m.timing);
        }
      };
      this.worker.onerror = (e) => {
        this.error = e.message || 'worker error';
        if (!settled) { settled = true; reject(new Error(this.error)); }
      };

      const y = this.cfg.yolo;
      this.worker.postMessage({
        type: 'init',
        baseUrl: new URL('../../', import.meta.url).href,
        modelCandidates: y.modelCandidates,
        providers: y.providers,
        preferredInputSize: y.preferredInputSize,
        keepClasses: y.keepClasses,
        confThreshold: y.confThreshold,
        iouThreshold: y.iouThreshold,
      });
    });
  }

  /** 是否該提交這一幀（受節流與 in-flight 限制） */
  shouldSubmit(now) {
    if (!this.ready || this.inFlight) return false;
    const minGap = 1000 / this.cfg.loop.detectHz;
    return now - this.lastSubmitTs >= minGap;
  }

  /**
   * 提交一幀。source 可以是 HTMLVideoElement / ImageBitmap / canvas。
   * 用 createImageBitmap 轉成可 transfer 的物件，零複製丟進 worker。
   */
  async submit(source, now) {
    if (!this.shouldSubmit(now)) return false;
    this.inFlight = true;
    this.lastSubmitTs = now;
    try {
      const bitmap = await createImageBitmap(source);
      this.worker.postMessage({ type: 'frame', ts: now, bitmap }, [bitmap]);
      return true;
    } catch (e) {
      this.inFlight = false;
      console.warn('[A-Eye] createImageBitmap 失敗:', e.message);
      return false;
    }
  }

  /**
   * 離線回放用：同步等待這一幀的偵測結果。
   * 即時模式絕不該用這個（會把主執行緒綁住），但離線評測需要
   * 「每一步都確實跑過偵測」的決定性行為，不能掉幀。
   */
  async runOnce(source, ts) {
    if (!this.ready) return [];
    const bitmap = await createImageBitmap(source);
    return new Promise((resolve) => {
      const prev = this.onResult;
      this.onResult = (boxes, rts, timing) => {
        this.onResult = prev;
        if (timing) this.timing = timing;
        resolve(boxes);
      };
      this.inFlight = true;
      this.worker.postMessage({ type: 'frame', ts, bitmap }, [bitmap]);
    });
  }

  dispose() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.ready = false;
    // 必須清掉 inFlight：worker 被 terminate 後那筆請求永遠不會回來，
    // 若留著 true，重新啟動後 shouldSubmit() 會永遠回 false（整個偵測靜默）
    this.inFlight = false;
    this.lastSubmitTs = 0;
    this.info = null;
    this.error = null;
    this.onResult = null;
  }
}
