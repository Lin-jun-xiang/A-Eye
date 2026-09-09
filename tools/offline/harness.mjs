// =============================================
// 離線 pipeline 跑機（node，不需要瀏覽器）
// =============================================
// 為什麼需要它：在這之前每一輪除錯都得「使用者用手機測 → 貼面板截圖」，
// 一次往返成本很高，而且只看得到當下那一格數字。有了這個跑機，
// 同一支影片可以在每次改動後重跑，內部狀態全部印出來。
//
// 它與 replay.html / analyze.html 跑的是**同一份 pipeline**，差別只在
// 執行環境：這裡用 onnxruntime-node 跑 YOLO、用 node 載入 opencv.js。
//
// 前置作業（都不進 repo，避免讓這個專案背上 npm 依賴）：
//   cd tools/offline
//   npm init -y && npm i onnxruntime-node
//   curl -o opencv.js https://docs.opencv.org/4.9.0/opencv.js
//
// 用法見同目錄的 README.md。
//
// ---- 踩過的三個坑（都在這裡留下註解，因為它們都會靜默地無限等待）----
//  1. `await cv`：opencv.js 的 Module 有一個相容性用的 then()，它用 Module
//     自己 resolve —— 而 Module 又是 thenable，Promise 機制因此無限遞迴。
//  2. 同一個坑的第二個化身：async function 裡 `return cv` 也會觸發同樣的遞迴。
//  3. 只掛 onRuntimeInitialized 不夠：runtime 可能在掛上 handler 之前就好了。
import { createRequire } from 'module';
import path from 'path';

const require_ = createRequire(import.meta.url);

// ---------- 1) 瀏覽器 API 的最小 shim ----------
// pipeline 內部只用到 canvas 的 drawImage / getImageData，
// 而 cv.imread 會用 instanceof HTMLCanvasElement / HTMLImageElement 檢查型別。
class ImageDataShim {
  constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
}
class Ctx2D {
  constructor(canvas) { this.canvas = canvas; this.fillStyle = '#000'; }
  fillRect(x, y, w, h) {
    const c = this.canvas;
    for (let yy = y; yy < y + h && yy < c.height; yy++) {
      for (let xx = x; xx < x + w && xx < c.width; xx++) {
        const i = (yy * c.width + xx) * 4;
        c.data[i] = 0; c.data[i + 1] = 0; c.data[i + 2] = 0; c.data[i + 3] = 255;
      }
    }
  }
  /** 支援 drawImage(src, sx,sy,sw,sh, dx,dy,dw,dh) 與 (src, dx,dy,dw,dh)，最近鄰縮放 */
  drawImage(src, ...a) {
    let sx = 0, sy = 0, sw = src.width, sh = src.height, dx = 0, dy = 0, dw, dh;
    if (a.length === 8) { [sx, sy, sw, sh, dx, dy, dw, dh] = a; }
    else if (a.length === 4) { [dx, dy, dw, dh] = a; }
    else { dw = src.width; dh = src.height; }
    const c = this.canvas;
    for (let y = 0; y < dh; y++) {
      const ty = dy + y;
      if (ty < 0 || ty >= c.height) continue;
      const syy = Math.min(src.height - 1, Math.max(0, Math.round(sy + (y + 0.5) * sh / dh - 0.5)));
      for (let x = 0; x < dw; x++) {
        const tx = dx + x;
        if (tx < 0 || tx >= c.width) continue;
        const sxx = Math.min(src.width - 1, Math.max(0, Math.round(sx + (x + 0.5) * sw / dw - 0.5)));
        const si = (syy * src.width + sxx) * 4, di = (ty * c.width + tx) * 4;
        c.data[di] = src.data[si];
        c.data[di + 1] = src.data[si + 1];
        c.data[di + 2] = src.data[si + 2];
        c.data[di + 3] = 255;
      }
    }
  }
  getImageData(x, y, w, h) {
    const c = this.canvas;
    const out = new Uint8ClampedArray(w * h * 4);
    for (let yy = 0; yy < h; yy++) {
      const src = ((y + yy) * c.width + x) * 4;
      out.set(c.data.subarray(src, src + w * 4), yy * w * 4);
    }
    return new ImageDataShim(out, w, h);
  }
}
class CanvasShim {
  constructor(w = 1, h = 1) { this._w = w; this._h = h; this.data = new Uint8ClampedArray(w * h * 4); }
  get width() { return this._w; }
  set width(v) { if (v !== this._w) { this._w = v; this._alloc(); } }
  get height() { return this._h; }
  set height(v) { if (v !== this._h) { this._h = v; this._alloc(); } }
  _alloc() { this.data = new Uint8ClampedArray(Math.max(1, this._w * this._h * 4)); }
  getContext() { return this._ctx || (this._ctx = new Ctx2D(this)); }
}

export function installShims() {
  globalThis.HTMLCanvasElement = CanvasShim;
  globalThis.HTMLImageElement = class {};
  globalThis.HTMLVideoElement = class {};
  globalThis.ImageData = ImageDataShim;
  globalThis.document = { createElement: (t) => (t === 'canvas' ? new CanvasShim() : {}) };
}

// ---------- 2) 載入 opencv.js ----------
export async function loadCv(cvPath) {
  const cv = require_(path.resolve(cvPath));
  globalThis.cv = cv;                 // 不能 `await cv`，見檔頭的坑 1
  if (!cv.Mat) {
    await new Promise((res, rej) => {
      const t0 = Date.now();
      cv.onRuntimeInitialized = () => res();
      const poll = () => {                                   // 見檔頭的坑 3
        if (cv.Mat) return res();
        if (Date.now() - t0 > 120000) return rej(new Error('opencv.js runtime 逾時'));
        setTimeout(poll, 100);
      };
      poll();
    });
  }
  return true;                        // 不能 return cv，見檔頭的坑 2
}

// ---------- 3) 偵測器（與 detector.worker.js 相同的 letterbox / 解碼 / NMS）----------
export class NodeDetector {
  constructor(cfg, modelPath, ort) { this.cfg = cfg; this.modelPath = modelPath; this.ort = ort; this.size = 640; }
  async init() {
    this.sess = await this.ort.InferenceSession.create(this.modelPath);
    const meta = this.sess.inputMetadata && this.sess.inputMetadata[0];
    if (meta && meta.shape && typeof meta.shape[2] === 'number') this.size = meta.shape[2];
    this.inputName = this.sess.inputNames[0];
    this.outputName = this.sess.outputNames[0];
    return { model: path.basename(this.modelPath), provider: 'node-cpu', inputSize: this.size };
  }
  async detect(img) {
    const S = this.size, n = S * S;
    const scale = Math.min(S / img.width, S / img.height);
    const nw = Math.round(img.width * scale), nh = Math.round(img.height * scale);
    const dx = Math.round((S - nw) / 2), dy = Math.round((S - nh) / 2);
    const t = new Float32Array(3 * n);
    for (let y = 0; y < nh; y++) {
      const sy = Math.min(img.height - 1, Math.round(y / scale));
      for (let x = 0; x < nw; x++) {
        const sx = Math.min(img.width - 1, Math.round(x / scale));
        const si = (sy * img.width + sx) * 4, di = (y + dy) * S + (x + dx);
        t[di] = img.data[si] / 255;
        t[n + di] = img.data[si + 1] / 255;
        t[2 * n + di] = img.data[si + 2] / 255;
      }
    }
    const feeds = {}; feeds[this.inputName] = new this.ort.Tensor('float32', t, [1, 3, S, S]);
    const out = await this.sess.run(feeds);
    return this._decode(out[this.outputName], { scale, dx, dy, vw: img.width, vh: img.height });
  }
  _decode(o, geo) {
    const y = this.cfg.yolo, data = o.data, numDet = o.dims[2], boxes = [];
    for (let i = 0; i < numDet; i++) {
      let best = 0, bestCls = -1;
      for (const c of y.keepClasses) {
        const sc = data[(4 + c) * numDet + i];
        if (sc > best) { best = sc; bestCls = c; }
      }
      if (bestCls < 0 || best < y.confThreshold) continue;
      const cx = data[i], cy = data[numDet + i];
      const w = data[2 * numDet + i], h = data[3 * numDet + i];
      let x = (cx - w / 2 - geo.dx) / geo.scale, yy = (cy - h / 2 - geo.dy) / geo.scale;
      let bw = w / geo.scale, bh = h / geo.scale;
      if (x < 0) { bw += x; x = 0; }
      if (yy < 0) { bh += yy; yy = 0; }
      if (x + bw > geo.vw) bw = geo.vw - x;
      if (yy + bh > geo.vh) bh = geo.vh - yy;
      if (bw <= 1 || bh <= 1) continue;
      boxes.push({ x, y: yy, w: bw, h: bh, score: best, classId: bestCls });
    }
    const iou = (a, b) => {
      const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
      const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
      const iw = x2 - x1, ih = y2 - y1;
      if (iw <= 0 || ih <= 0) return 0;
      const inter = iw * ih;
      return inter / (a.w * a.h + b.w * b.h - inter);
    };
    boxes.sort((a, b) => b.score - a.score);
    const keep = [], dead = new Uint8Array(boxes.length);
    for (let i = 0; i < boxes.length; i++) {
      if (dead[i]) continue;
      keep.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (dead[j] || boxes[i].classId !== boxes[j].classId) continue;
        if (iou(boxes[i], boxes[j]) > y.iouThreshold) dead[j] = 1;
      }
    }
    return keep;
  }
}

// ---------- 4) 從 stdin 讀 rawvideo 幀 ----------
// 用串流而不是先存成檔案：一段 18 秒的 588x940 rawvideo 就有 400MB，
// 而且長影片會直接爆掉。
export function frameReader(stream, frameBytes) {
  let buf = Buffer.alloc(0);
  let done = false;
  const waiters = [];
  stream.on('data', (c) => {
    buf = buf.length ? Buffer.concat([buf, c]) : c;
    while (waiters.length && buf.length >= frameBytes) {
      const f = buf.subarray(0, frameBytes);
      buf = buf.subarray(frameBytes);
      waiters.shift()(f);
    }
  });
  stream.on('end', () => { done = true; while (waiters.length) waiters.shift()(null); });
  return () => new Promise((res) => {
    if (buf.length >= frameBytes) {
      const f = buf.subarray(0, frameBytes);
      buf = buf.subarray(frameBytes);
      return res(f);
    }
    if (done) return res(null);
    waiters.push(res);
  });
}
