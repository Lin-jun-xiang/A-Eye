// =============================================
// YOLO 推論 Worker（classic worker，用 importScripts 載入 ORT）
// =============================================
// 為什麼要放 Worker：
//   v6 把 YOLO / MiDaS / UFLD 全部 await 在主執行緒的同一個迴圈裡，
//   於是「YOLO 推論耗掉的幾百毫秒」直接變成整個系統的節拍，
//   光流拿到的 bbox 與它自己抓的影像相差好幾百毫秒（時間錯位），
//   而 LK 光流的 small-motion 假設在 dt = 0.5~1s 下完全崩潰。
//   把推論搬到 Worker 後，主執行緒可以用 video 的原生幀率跑光流與追蹤，
//   偵測結果帶著自己的時間戳非同步回來，由 KF 做時間對齊。

let ort = null;
let session = null;
let inputName = null;
let outputName = null;
let inputSize = 640;
let modelUrl = '';
let provider = '';
let keepClasses = [2, 5, 7, 9];
let confThreshold = 0.3;
let iouThreshold = 0.45;

let canvas = null;
let ctx = null;

const ORT_VERSION = '1.20.1';
// onnxruntime-web 不同版本的 bundle 檔名不完全一致，依序嘗試。
// ort.all.* 含 webgpu + wasm；退回 ort.min.js（wasm）。
const ORT_CANDIDATES = [
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.all.min.js`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`,
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js',
];

function loadOrt() {
  const errors = [];
  for (const url of ORT_CANDIDATES) {
    try {
      importScripts(url);
      if (typeof self.ort !== 'undefined') {
        ort = self.ort;
        const base = url.slice(0, url.lastIndexOf('/') + 1);
        try {
          ort.env.wasm.wasmPaths = base;
          ort.env.wasm.numThreads = Math.min(4, self.navigator?.hardwareConcurrency || 2);
          ort.env.wasm.simd = true;
          ort.env.logLevel = 'error';
        } catch (_) { /* 舊版可能沒有某些欄位 */ }
        return url;
      }
    } catch (e) {
      errors.push(`${url}: ${e.message}`);
    }
  }
  throw new Error('無法載入 onnxruntime-web\n' + errors.join('\n'));
}

async function createSession(candidates, providers, preferredSize) {
  const errors = [];
  for (const url of candidates) {
    for (const ep of providers) {
      try {
        const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
        const s = await ort.InferenceSession.create(url, opts);
        const name = s.inputNames[0];
        // 暖機 + 尺寸探測：失敗就換下一個 EP / 模型，不要留下壞掉的 session
        const size = await probeInputSize(s, name, preferredSize);
        session = s;
        modelUrl = url;
        provider = ep;
        inputName = name;
        outputName = s.outputNames[0];
        inputSize = size;
        return;
      } catch (e) {
        errors.push(`${url.split('/').pop()} @ ${ep}: ${e.message}`);
      }
    }
  }
  throw new Error('所有模型 / EP 組合都失敗\n' + errors.join('\n'));
}

/**
 * 決定實際要餵的輸入尺寸。
 *
 * 模型可能是靜態軸（匯出時固定 640）或動態軸。ORT-web 的 inputMetadata
 * 欄位格式跨版本不一致，讀不到時猜錯尺寸會在第一次推論就爆 shape mismatch。
 * 所以這裡不只讀 metadata，還用「零張量暖機推論」實際試出可用的尺寸：
 *   1. 先試 metadata 讀到的（或偏好的）尺寸
 *   2. 失敗就依序試常見尺寸
 * 暖機同時也把計算圖預先編譯好 —— 第一次推論本來就是最慢的一次，
 * 讓它發生在啟動階段而不是使用者眼前的第一幀。
 */
function metadataInputSize(s, name) {
  try {
    const meta = s.inputMetadata;
    const dims = Array.isArray(meta)
      ? meta.find((m) => m.name === name)?.shape
      : meta?.[name]?.dimensions;
    if (dims && dims.length === 4) {
      const h = dims[2];
      if (typeof h === 'number' && h > 0) return h;   // 靜態軸
    }
  } catch (_) { /* 取不到就交給暖機去試 */ }
  return null;
}

async function probeInputSize(s, name, preferred) {
  const fromMeta = metadataInputSize(s, name);
  const candidates = [...new Set([fromMeta, preferred, 640, 480, 384, 320].filter(Boolean))];
  const errors = [];
  for (const size of candidates) {
    try {
      const zeros = new Float32Array(3 * size * size);
      const feeds = {};
      feeds[name] = new ort.Tensor('float32', zeros, [1, 3, size, size]);
      const out = await s.run(feeds);
      // 順便確認輸出形狀是 YOLOv8 的 [1, 4+C, N]
      const o = out[s.outputNames[0]];
      if (o && o.dims && o.dims.length === 3 && o.dims[1] >= 5) return size;
      errors.push(`${size}: 輸出形狀非預期 ${o && o.dims}`);
    } catch (e) {
      errors.push(`${size}: ${e.message}`);
    }
  }
  throw new Error('找不到可用的輸入尺寸\n' + errors.join('\n'));
}

function ensureCanvas(size) {
  if (!canvas || canvas.width !== size) {
    canvas = new OffscreenCanvas(size, size);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
}

/** letterbox 前處理：等比縮放 + 補黑邊，回傳 NCHW float32 [0,1] */
function preprocess(bitmap) {
  const size = inputSize;
  ensureCanvas(size);
  const vw = bitmap.width, vh = bitmap.height;
  const scale = Math.min(size / vw, size / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const dx = Math.round((size - nw) / 2), dy = Math.round((size - nh) / 2);

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, 0, 0, vw, vh, dx, dy, nw, nh);

  const px = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = px[i * 4] / 255;
    out[n + i] = px[i * 4 + 1] / 255;
    out[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return { tensor: out, scale, dx, dy, vw, vh };
}

/**
 * 解碼 YOLOv8 輸出 [1, 4+80, N]。
 * 關鍵優化：只對 keepClasses 取 max，而不是全部 80 類。
 * 8400 × 80 = 672k 次比較 → 8400 × 4 = 34k，解碼快 ~20 倍。
 */
function decode(output, geo) {
  const data = output.data;
  const dims = output.dims;
  const numDet = dims[2];
  const stride = dims[1];      // 4 + numClasses
  const boxes = [];

  for (let i = 0; i < numDet; i++) {
    let best = 0, bestCls = -1;
    for (let k = 0; k < keepClasses.length; k++) {
      const c = keepClasses[k];
      const sc = data[(4 + c) * numDet + i];
      if (sc > best) { best = sc; bestCls = c; }
    }
    if (bestCls < 0 || best < confThreshold) continue;

    const cx = data[i];
    const cy = data[numDet + i];
    const w = data[2 * numDet + i];
    const h = data[3 * numDet + i];

    // letterbox -> 原始影像座標
    let x = (cx - w / 2 - geo.dx) / geo.scale;
    let y = (cy - h / 2 - geo.dy) / geo.scale;
    let bw = w / geo.scale;
    let bh = h / geo.scale;

    if (x < 0) { bw += x; x = 0; }
    if (y < 0) { bh += y; y = 0; }
    if (x + bw > geo.vw) bw = geo.vw - x;
    if (y + bh > geo.vh) bh = geo.vh - y;
    if (bw <= 1 || bh <= 1) continue;

    boxes.push({ x, y, w: bw, h: bh, score: best, classId: bestCls });
  }
  if (stride < 4 + 1) return [];
  return nms(boxes);
}

function iouXYWH(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = x2 - x1, ih = y2 - y1;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  const dead = new Uint8Array(boxes.length);
  for (let i = 0; i < boxes.length; i++) {
    if (dead[i]) continue;
    keep.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (dead[j] || boxes[i].classId !== boxes[j].classId) continue;
      if (iouXYWH(boxes[i], boxes[j]) > iouThreshold) dead[j] = 1;
    }
  }
  return keep;
}

self.onmessage = async (ev) => {
  const msg = ev.data;

  if (msg.type === 'init') {
    try {
      const ortUrl = loadOrt();
      keepClasses = msg.keepClasses || keepClasses;
      confThreshold = msg.confThreshold ?? confThreshold;
      iouThreshold = msg.iouThreshold ?? iouThreshold;
      const urls = msg.modelCandidates.map((p) => new URL(p, msg.baseUrl).href);
      await createSession(urls, msg.providers, msg.preferredInputSize);
      self.postMessage({
        type: 'ready',
        model: modelUrl.split('/').pop(),
        provider,
        inputSize,
        ortUrl: ortUrl.split('/').pop(),
      });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message });
    }
    return;
  }

  if (msg.type === 'frame') {
    if (!session) {
      msg.bitmap?.close?.();
      self.postMessage({ type: 'result', ts: msg.ts, boxes: [], dropped: true });
      return;
    }
    const t0 = performance.now();
    try {
      const geo = preprocess(msg.bitmap);
      msg.bitmap.close();
      const tPre = performance.now();
      const feeds = {};
      feeds[inputName] = new ort.Tensor('float32', geo.tensor, [1, 3, inputSize, inputSize]);
      const res = await session.run(feeds);
      const tInfer = performance.now();
      const boxes = decode(res[outputName], geo);
      const tDecode = performance.now();
      self.postMessage({
        type: 'result',
        ts: msg.ts,
        boxes,
        timing: {
          pre: tPre - t0,
          infer: tInfer - tPre,
          decode: tDecode - tInfer,
          total: tDecode - t0,
        },
      });
    } catch (e) {
      msg.bitmap?.close?.();
      self.postMessage({ type: 'result', ts: msg.ts, boxes: [], error: e.message });
    }
  }
};
