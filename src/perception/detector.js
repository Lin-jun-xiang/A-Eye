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

  /** onnxruntime-web 的 CDN 候選網址 */
  ortUrls() {
    const y = this.cfg.yolo;
    const base = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${y.ortVersion}/dist/`;
    return y.ortFiles.map((f) => base + f);
  }

  /**
   * 把 ORT 抓下來轉成「同源 blob URL」。
   *
   * 為什麼要多這一步：worker 裡 importScripts() 一個跨來源網址，在 WebKit 上
   * 會因為 Service Worker 把回應變成 opaque 而失敗（規範禁止 importScripts
   * 接受 opaque response），錯誤訊息是「Network response is CORS-cross-origin」。
   * 我們已經修好 sw.js 不再攔截跨來源，但使用者裝置上可能還留著舊的 SW，
   * 所以這裡再加一層：主執行緒用 fetch()（走正常 CORS，jsDelivr 有送
   * access-control-allow-origin: *）抓下來，轉成同源 blob，worker 載入 blob
   * 就完全不涉及跨來源了。
   *
   * 注意 wasm 仍然從 CDN 抓 —— 那是 ORT 自己用 fetch() 發的 CORS 請求，
   * 不受 importScripts 的限制，所以 baseUrl 一定要指回 CDN。
   */
  async _prepareOrt() {
    const errors = [];
    for (const url of this.ortUrls()) {
      try {
        const res = await fetch(url, { mode: 'cors', credentials: 'omit' });
        if (!res.ok) { errors.push(`${url}: HTTP ${res.status}`); continue; }
        const blob = await res.blob();
        return {
          blobUrl: URL.createObjectURL(blob),
          baseUrl: url.slice(0, url.lastIndexOf('/') + 1),
          sourceUrl: url,
        };
      } catch (e) {
        errors.push(`${url}: ${e.message}`);
      }
    }
    console.warn('[A-Eye] ORT 預先下載失敗，改由 worker 直接載入：\n' + errors.join('\n'));
    return null;
  }

  async init() {
    const prepared = await this._prepareOrt();
    this._ortBlobUrl = prepared ? prepared.blobUrl : null;

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
      const fam = this.cfg.detector.family;
      const d = this.cfg.detr;
      this.worker.postMessage({
        type: 'init',
        baseUrl: new URL('../../', import.meta.url).href,
        ortBlobUrl: prepared ? prepared.blobUrl : null,
        ortBaseUrl: prepared ? prepared.baseUrl : null,
        ortSourceUrl: prepared ? prepared.sourceUrl : null,
        ortFallbackUrls: this.ortUrls(),
        family: fam,
        detr: d,
        modelCandidates: fam === 'detr' ? d.modelCandidates : y.modelCandidates,
        providers: y.providers,
        preferredInputSize: y.preferredInputSize,
        keepClasses: y.keepClasses,
        confThreshold: y.confThreshold,
        confLow: y.confLow,
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
    if (this._ortBlobUrl) { URL.revokeObjectURL(this._ortBlobUrl); this._ortBlobUrl = null; }
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
