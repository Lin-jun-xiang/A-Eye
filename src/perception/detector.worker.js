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

// 整個檔案包在 IIFE 裡，這不是風格偏好而是必要的：
// importScripts() 載入的腳本是在 worker 的「全域範疇」執行的，所以本檔案
// 任何頂層 let/const/function 都會與函式庫的全域宣告搶同一個命名空間。
// 我們原本在頂層宣告了 `let ort`，而 onnxruntime-web 的 UMD bundle 也在頂層
// 宣告 ort，於是 WebKit 直接丟出
//     SyntaxError: Can't create duplicate variable: 'ort'
// 腳本連執行的機會都沒有。包進 IIFE 後我們的宣告全都是函式範疇，
// 之後換任何函式庫都不會再撞名。
(function () {
'use strict';

let ortApi = null;
let session = null;
let inputName = null;
let outputName = null;
let inputSize = 640;
let modelUrl = '';
let provider = '';
let keepClasses = [2, 5, 7, 9];
let confThreshold = 0.3;
// BYTE 兩段式關聯的低分層門檻：解碼保留到這裡，由 tracker 分兩段使用。
// 低分框只能延續既有 track，不能建立新 track（見 config.js 的說明）。
let confLow = 0.1;
// 偵測器家族：'yolo' | 'detr'。兩者的前處理、輸出格式、類別編碼都不同，
// 但**對外的契約完全一樣**：回傳 app 內部（COCO-80）類別 id 的框陣列，
// 所以 tracker / frontCar / 一切下游都不需要知道用的是哪一個模型。
let family = 'yolo';
let detrCfg = null;
let iouThreshold = 0.45;

let canvas = null;
let ctx = null;

/**
 * 載入 onnxruntime-web。
 *
 * 優先順序：
 *   1. 主執行緒預先抓好的「同源 blob URL」
 *   2. 直接 importScripts CDN（僅在 blob 那條路失敗時）
 *
 * 為什麼不直接 importScripts CDN：
 *   worker 的 importScripts() 若被 Service Worker 攔截，SW 用 fetch() 重發
 *   跨來源 no-cors 請求後拿到的是 opaque response，而 HTML 規範禁止
 *   importScripts 接受 SW 提供的 opaque response —— WebKit 會丟出
 *   「Network response is CORS-cross-origin」，即使 CDN 本身有送
 *   access-control-allow-origin: *。改由主執行緒 fetch 成 blob 後就完全同源。
 *
 * wasmPaths 必須指回 CDN：那是 ORT 自己用 fetch() 發的 CORS 請求，
 * 不受 importScripts 的限制。blob URL 沒有可用的相對基底，不能拿來當 base。
 */
function configureOrt(baseUrl) {
  ortApi = self.ort;
  try {
    ortApi.env.wasm.wasmPaths = baseUrl;
    ortApi.env.wasm.numThreads = Math.min(4, self.navigator?.hardwareConcurrency || 2);
    ortApi.env.wasm.simd = true;
    ortApi.env.logLevel = 'error';
  } catch (_) { /* 舊版可能沒有某些欄位 */ }
}

function loadOrt(msg) {
  const attempts = [];
  if (msg.ortBlobUrl && msg.ortBaseUrl) {
    attempts.push({
      url: msg.ortBlobUrl,
      base: msg.ortBaseUrl,
      label: 'blob:' + (msg.ortSourceUrl || '').split('/').pop(),
    });
  }
  for (const u of msg.ortFallbackUrls || []) {
    attempts.push({ url: u, base: u.slice(0, u.lastIndexOf('/') + 1), label: u.split('/').pop() });
  }

  const errors = [];
  for (const a of attempts) {
    try {
      // 上一次嘗試可能已經把 ort 定義進全域了（即使它之後才拋錯）。
      // 若不先檢查就再 importScripts 一次，會撞上 duplicate variable 而
      // 掩蓋掉真正的錯誤。
      if (typeof self.ort === 'undefined') importScripts(a.url);
      if (typeof self.ort !== 'undefined') {
        configureOrt(a.base);
        return a.label;
      }
      errors.push(`${a.label}: 載入成功但找不到全域 ort`);
    } catch (e) {
      errors.push(`${a.label}: ${e.message}`);
    }
  }
  throw new Error(
    '無法載入 onnxruntime-web\n' + errors.join('\n')
    + '\n（跨來源 importScripts 在 WebKit 上本來就常被拒絕，'
    + '正常路徑是主執行緒預抓的 blob；若 blob 那條也失敗，看它的錯誤訊息。）'
  );
}

async function createSession(candidates, providers, preferredSize) {
  const errors = [];
  for (const url of candidates) {
    for (const ep of providers) {
      try {
        const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
        const s = await ortApi.InferenceSession.create(url, opts);
        const name = s.inputNames[0];
        let size;
        if (family === 'detr') {
          // DETR 的輸入是動態軸，尺寸由 detr.shortSide 決定，不需要探測。
          // 但仍然跑一次暖機推論：第一次推論本來就是最慢的一次，
          // 讓它發生在啟動階段而不是使用者眼前的第一幀。
          size = detrCfg.shortSide;
          await runDetr(s, new Float32Array(3 * size * size), size, size);
        } else {
          // 暖機 + 尺寸探測：失敗就換下一個 EP / 模型，不要留下壞掉的 session
          size = await probeInputSize(s, name, preferredSize);
        }
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
      feeds[name] = new ortApi.Tensor('float32', zeros, [1, 3, size, size]);
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

  // letterbox 的填充色**必須是灰 114**，不是黑。
  // Ultralytics 訓練與推論時用的都是 (114,114,114)，填黑等於在模型輸入裡
  // 塞進一塊訓練時沒見過的東西 —— 而直式手機畫面塞進 640x640 時，
  // 黑邊佔了輸入面積的 44%。
  // 實測（2026-09-10 夜間停等車陣，64 幀，同一個模型只改填充色）：
  //   黑填充  找到前車 72%，信心中位 0.20，≥0.30 的幀 19%
  //   灰 114  找到前車 77%，信心中位 0.26，≥0.30 的幀 36%
  // 什麼都沒換，通過門檻的幀數幾乎翻倍。
  ctx.fillStyle = 'rgb(114,114,114)';
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
    if (bestCls < 0 || best < Math.min(confLow, confThreshold)) continue;

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

/**
 * DETR 的前處理：短邊縮到 shortSide、ImageNet 標準化。
 *
 * 注意這裡**沒有 letterbox** —— DETR 吃任意長寬比的輸入，
 * 所以 YOLO 那個「填充色必須是灰 114」的坑在這條路上根本不存在，
 * 也不會浪費 44% 的輸入面積在黑邊上。
 */
function preprocessDetr(bitmap) {
  const d = detrCfg;
  const vw = bitmap.width, vh = bitmap.height;
  const sc = d.shortSide / Math.min(vw, vh);
  const nw = Math.round(vw * sc), nh = Math.round(vh * sc);
  if (!canvas || canvas.width !== nw || canvas.height !== nh) {
    canvas = new OffscreenCanvas(nw, nh);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
  ctx.drawImage(bitmap, 0, 0, vw, vh, 0, 0, nw, nh);
  const px = ctx.getImageData(0, 0, nw, nh).data;
  const n = nw * nh;
  const out = new Float32Array(3 * n);
  const m = d.mean, sd = d.std;
  for (let i = 0; i < n; i++) {
    out[i]         = (px[i * 4]     / 255 - m[0]) / sd[0];
    out[n + i]     = (px[i * 4 + 1] / 255 - m[1]) / sd[1];
    out[2 * n + i] = (px[i * 4 + 2] / 255 - m[2]) / sd[2];
  }
  return { tensor: out, nw, nh, vw, vh };
}

/** 跑一次 DETR（pixel_values + 固定形狀的 pixel_mask） */
function runDetr(sess, tensor, nh, nw) {
  const ms = detrCfg.maskSize;
  return sess.run({
    pixel_values: new ortApi.Tensor('float32', tensor, [1, 3, nh, nw]),
    // 這份匯出把 pixel_mask 固定成 [1,64,64]，模型內部會插值到特徵圖尺寸。
    // 我們不做 padding，遮罩全 1，所以尺寸不影響結果。
    pixel_mask: new ortApi.Tensor('int64', new BigInt64Array(ms * ms).fill(1n), [1, ms, ms]),
  });
}

/**
 * DETR 的解碼。與 YOLO 有三個結構性差異：
 *   1. 對 92 類做 softmax，**最後一類是「無物件」** —— 這正是它不會把
 *      方向盤硬塞進某個類別的原因（實測內裝假框 0%，YOLO 是 25~33% 的幀）
 *   2. 框是**正規化的 cxcywh**（相對整張圖），不需要 letterbox 反算
 *   3. 集合預測 + 二分圖匹配 → 一個物件只對應一個 query，**不需要 NMS**
 *      （也就不會出現把前車和旁車併成一個大框的情形）
 */
function decodeDetr(res, geo) {
  const d = detrCfg;
  const lg = res.logits, bx = res.pred_boxes;
  const Q = lg.dims[1], C = lg.dims[2];
  const boxes = [];
  for (let q = 0; q < Q; q++) {
    const off = q * C;
    let mx = -Infinity;
    for (let c = 0; c < C; c++) { const v = lg.data[off + c]; if (v > mx) mx = v; }
    let sum = 0;
    for (let c = 0; c < C; c++) sum += Math.exp(lg.data[off + c] - mx);
    // 最後一類（no-object）不參與比大小，但**留在 softmax 的分母裡** ——
    // 它就是「這個 query 什麼都不是」的機率質量，也是假框歸零的機制
    let best = -1, bs = 0;
    for (let c = 0; c < C - 1; c++) {
      const pr = Math.exp(lg.data[off + c] - mx) / sum;
      if (pr > bs) { bs = pr; best = c; }
    }
    const mapped = d.classMap[best];
    // 與 YOLO 那條路一致：解碼保留到**低分層**，由 tracker 分兩段使用。
    // 少了這一行，BYTE 在 DETR 上完全不會運作（實測面板：低分框 0 → 救回 0），
    // 而 DETR 的推論慢、偵測率低，正是最需要靠低分框把軌跡接起來的情況。
    if (mapped === undefined || bs < Math.min(confLow, confThreshold)) continue;
    const cx = bx.data[q * 4] * geo.vw, cy = bx.data[q * 4 + 1] * geo.vh;
    let bw = bx.data[q * 4 + 2] * geo.vw, bh = bx.data[q * 4 + 3] * geo.vh;
    let x = cx - bw / 2, y = cy - bh / 2;
    if (x < 0) { bw += x; x = 0; }
    if (y < 0) { bh += y; y = 0; }
    if (x + bw > geo.vw) bw = geo.vw - x;
    if (y + bh > geo.vh) bh = geo.vh - y;
    if (bw <= 1 || bh <= 1) continue;
    boxes.push({ x, y, w: bw, h: bh, score: bs, classId: mapped });
  }
  return boxes;                     // 刻意不做 NMS
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
      const ortUrl = loadOrt(msg);
      family = msg.family || 'yolo';
      detrCfg = msg.detr || null;
      keepClasses = msg.keepClasses || keepClasses;
      confThreshold = (family === 'detr' ? detrCfg.confThreshold : msg.confThreshold)
        ?? confThreshold;
      confLow = msg.confLow ?? confLow;
      iouThreshold = msg.iouThreshold ?? iouThreshold;
      const urls = msg.modelCandidates.map((p) => new URL(p, msg.baseUrl).href);
      await createSession(urls, msg.providers, msg.preferredInputSize);
      self.postMessage({
        type: 'ready',
        model: modelUrl.split('/').pop(),
        family,
        provider,
        inputSize,
        ortUrl,   // loadOrt() 已回傳精簡標籤
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
      const isDetr = family === 'detr';
      const geo = isDetr ? preprocessDetr(msg.bitmap) : preprocess(msg.bitmap);
      msg.bitmap.close();
      const tPre = performance.now();
      let res;
      if (isDetr) {
        res = await runDetr(session, geo.tensor, geo.nh, geo.nw);
      } else {
        const feeds = {};
        feeds[inputName] = new ortApi.Tensor('float32', geo.tensor, [1, 3, inputSize, inputSize]);
        res = await session.run(feeds);
      }
      const tInfer = performance.now();
      const boxes = isDetr ? decodeDetr(res, geo) : decode(res[outputName], geo);
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
})();
