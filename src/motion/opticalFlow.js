// =============================================
// 光流量測（OpenCV.js LK）
// =============================================
// 相對 v6 的四個結構性改動：
//
// 1. 【原生解析度 ROI】v6 把整幀降到 320×180 才跑光流。估算一下：
//    60°HFOV、320 寬 → f ≈ 277px；10m 外車寬 1.8m → bbox 約 50px；
//    以 1 m/s 起步、dt=0.5s，bbox 寬 50→47.6px，邊緣點位移約 1.2px
//    —— 和 LK 的雜訊底同一量級，SNR < 1，真訊號直接沉掉。
//    改成「bbox 外擴後在原生解析度裁切」，同樣情境位移約 5px，SNR 提升 3~4 倍。
//
// 2. 【錨定 ROI】裁切框在兩次重撒點之間固定不動。若每幀跟著 bbox 移動，
//    座標系本身就在變，會憑空製造出尺度變化 —— 那是純粹的假訊號。
//
// 3. 【FB 一致性檢核】v6 只看 LK 的 status，追丟的點會滑到背景上繼續投票。
//    這裡做 forward-backward：再往回追一次，回不到原點的點直接剔除。
//    另外強制剔除跑出目標框的點。
//
// 4. 【輸出改為剛體參數，不是逐點票】對前景點與背景點分別擬合相似變換
//    （平移+旋轉+尺度）。前車是剛體，它的運動本來就只有這幾個自由度；
//    把 80 個高度相關的點濃縮成「1 個尺度參數 + 它的標準誤」才是正確的降維。
//    相對尺度 s_rel = s_fg / s_bg 還順便消掉了全域縮放（對焦呼吸、輕微前進）。

import { median } from '../util/math.js';
import { CONFIG } from '../config.js';

/**
 * 擬合相似變換（4 自由度：平移 + 旋轉 + 等向尺度），並回報估計的不確定度。
 *
 * 為什麼自己算而不用 OpenCV：
 *   原本用 `cv.estimateAffinePartial2D`，但**官方 opencv.js 根本沒有編入這個函式**
 *   （opencv/opencv#20538，只有 6 自由度的 estimateAffine2D）。於是它是 undefined，
 *   呼叫就丟例外、被 try/catch 吃掉、回傳 null —— 每一次量測都是 fg-fit-fail。
 *   實車路測的除錯面板證實了這件事：`flow ok 0/225 | fg-fit-fail:41`，
 *   也就是尺度變化率這條路從 v7 寫出來就沒有運作過一次。
 *
 * 而且自己算其實更好：
 *   1. 4 自由度的相似變換有**閉式最小平方解**（Procrustes），不需要迭代或 RANSAC
 *   2. 這正是 σ_s 傳播公式假設的估計量；用 6 自由度的仿射會多兩個自由度的雜訊
 *   3. **決定性**（沒有 RANSAC 的隨機取樣）→ 離線回放評測可重現
 *   4. 純 JS、不依賴 cv → 可以在 node 裡做單元測試
 *
 * 推導：令 q = M·p + t，M = [[a, b], [-b, a]]（a = s·cosθ、b = s·sinθ）。
 * 對中心化座標最小化 Σ|q − M·p|²，兩個正規方程給出
 *   a =  Σ(dp·dq) / Σ|dp|²        （內積）
 *   b = −Σ(dp × dq) / Σ|dp|²      （外積）
 * 離群點用 MAD 估出的 σ 做硬性剔除後重擬合（IRLS 的簡化版），
 * 因為前景點已經先過 forward-backward 與「跑出目標框」兩道檢核。
 *
 * σ_s 的傳播：尺度是「以質心為原點的徑向縮放」，
 *   σ_s ≈ σ_residual / (rRms · √n)
 * 這是最小平方估計的標準誤，不是憑感覺的門檻。
 */
export function fitSimilarity(prev, next, cfg) {
  const n = prev.length / 2;
  if (n < 4) return null;
  const keep = new Uint8Array(n).fill(1);
  let a = 1, b = 0, tx = 0, ty = 0, cx = 0, cy = 0, nIn = n;
  let resid = [], sigmaResid = 0, rRms = 0;

  for (let iter = 0; iter < 3; iter++) {
    let px = 0, py = 0, qx = 0, qy = 0;
    nIn = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      px += prev[i * 2]; py += prev[i * 2 + 1];
      qx += next[i * 2]; qy += next[i * 2 + 1];
      nIn++;
    }
    if (nIn < 4) return null;
    px /= nIn; py /= nIn; qx /= nIn; qy /= nIn;

    let dot = 0, cross = 0, den = 0, sumR2 = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      const dpx = prev[i * 2] - px, dpy = prev[i * 2 + 1] - py;
      const dqx = next[i * 2] - qx, dqy = next[i * 2 + 1] - qy;
      dot += dpx * dqx + dpy * dqy;
      cross += dpx * dqy - dpy * dqx;
      den += dpx * dpx + dpy * dpy;
      sumR2 += dpx * dpx + dpy * dpy;
    }
    if (!(den > 1e-9)) return null;
    a = dot / den;
    b = -cross / den;
    cx = px; cy = py;
    tx = qx - (a * px + b * py);
    ty = qy - (-b * px + a * py);
    rRms = Math.sqrt(sumR2 / nIn);

    // 殘差 → MAD 尺度 → 剔除離群點（最後一輪不再剔除，直接用來報告）
    resid = [];
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      const x = prev[i * 2], y = prev[i * 2 + 1];
      const ex = a * x + b * y + tx;
      const ey = -b * x + a * y + ty;
      resid.push(Math.hypot(next[i * 2] - ex, next[i * 2 + 1] - ey));
    }
    // 殘差「大小」是 Rayleigh 分布，不是零均值的常態分布 ——
    // 直接對大小取 MAD 會低估每軸的 σ 約 2.4 倍，後果有兩個：
    //   (a) 剔除門檻過嚴 → 好點被大量丟掉 → nIn 不足 → 擬合失敗
    //   (b) σ_s 低報 → SPRT 過度自信 → 實際誤警率遠高於設定的 α
    // Rayleigh 的中位數 = 1.1774σ，所以由中位數反推每軸 σ 才是對的。
    // 再乘上自由度校正 √(n/(n−4))：殘差是「扣掉 4 個已擬合參數之後」的剩餘，
    // 不校正的話 σ 會系統性低估（n=20 時約低 11%），SPRT 就會偏樂觀。
    const dof = nIn > 5 ? Math.sqrt(nIn / (nIn - 4)) : 1.5;
    sigmaResid = Math.max((median(resid) / 1.1774) * dof, 0.05);   // 下限防止除以 0
    if (iter === 2) break;
    let dropped = 0, k = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      if (resid[k++] > cfg.outlierSigma * sigmaResid) { keep[i] = 0; dropped++; }
    }
    if (!dropped) break;
  }

  const s = Math.hypot(a, b);
  if (!(s > 0.5 && s < 2.0)) return null;             // 離譜的解，視為擬合失敗
  const sigmaS = rRms > 1e-3 ? sigmaResid / (rRms * Math.sqrt(nIn)) : Infinity;

  return {
    a, b, tx, ty, s, theta: Math.atan2(-b, a),
    sigmaS, sigmaResid, rRms,
    n, nIn, inlierRatio: nIn / n,
    centroid: { x: cx, y: cy },
    apply: (x, y) => ({ x: a * x + b * y + tx, y: -b * x + a * y + ty }),
  };
}

/** cv.Mat 生命週期管理：一次 tick 內配置的都登記，結束一律釋放 */
class MatBag {
  constructor() { this.list = []; }
  add(m) { this.list.push(m); return m; }
  freeAll() {
    for (const m of this.list) {
      try { if (m && !m.isDeleted()) m.delete(); } catch (_) { /* ignore */ }
    }
    this.list.length = 0;
  }
}

export class OpticalFlow {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.cvReady = false;

    this.canvas = typeof document !== 'undefined'
      ? document.createElement('canvas')
      : new OffscreenCanvas(64, 64);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.prevGray = null;       // cv.Mat
    this.fgPts = null;          // [x0,y0,x1,y1,...] ROI 座標（目前位置）
    this.bgPts = null;
    // 量測基線的參考位置：與 fgPts / bgPts 同順序、同長度，
    // 記錄「這些點在 refTs 時刻的位置」。尺度是拿 ref → cur 擬合出來的，
    // 不是相鄰兩幀 —— 相鄰兩幀的尺度變化被雜訊淹沒。
    this.fgRef = null;
    this.bgRef = null;
    this.refTs = 0;
    this.anchor = null;         // { x, y, w, h } 影像座標（整數）
    this.roiScale = 1;
    this.roiW = 0; this.roiH = 0;
    this.lastSampleTs = 0;
    this.lastTs = 0;
    this.targetAtSample = null;
  }

  async loadCv(loadScript) {
    if (this.cvReady) return true;
    if (typeof cv === 'undefined') {
      const errors = [];
      let loaded = false;
      for (const url of this.cfg.flow.cvUrls) {
        try {
          await loadScript(url);
          loaded = true;
          break;
        } catch (e) {
          errors.push(`${url}: ${e.message}`);
        }
      }
      if (!loaded) {
        console.warn('[A-Eye] OpenCV.js 載入失敗:\n' + errors.join('\n'));
        return false;
      }
    }
    await new Promise((resolve) => {
      const check = () => {
        if (typeof cv !== 'undefined' && cv.Mat) resolve();
        else setTimeout(check, 50);
      };
      if (typeof cv !== 'undefined' && typeof cv.then === 'function') cv.then(() => resolve());
      else if (typeof cv !== 'undefined') { cv.onRuntimeInitialized = () => resolve(); check(); }
      else check();
    });
    this.cvReady = !!(typeof cv !== 'undefined' && cv.Mat);
    return this.cvReady;
  }

  reset() {
    if (this.prevGray) {
      try { if (!this.prevGray.isDeleted()) this.prevGray.delete(); } catch (_) {}
    }
    this.prevGray = null;
    this.fgPts = null;
    this.bgPts = null;
    this.fgRef = null;
    this.bgRef = null;
    this.refTs = 0;
    this.anchor = null;
    this.targetAtSample = null;
    this.lastSampleTs = 0;
    this.lastTs = 0;
  }

  get active() { return !!this.prevGray; }

  // ---------- ROI 幾何 ----------

  /** 依目標框決定錨定 ROI（影像座標，整數對齊避免重取樣抖動） */
  _computeAnchor(target, vw, vh) {
    const f = this.cfg.flow.roiInflate;
    const cx = target.x + target.w / 2, cy = target.y + target.h / 2;
    let w = target.w * f, h = target.h * f;
    let x = Math.round(cx - w / 2), y = Math.round(cy - h / 2);
    w = Math.round(w); h = Math.round(h);
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > vw) w = vw - x;
    if (y + h > vh) h = vh - y;
    return { x, y, w, h };
  }

  /** 目標是否還安穩地待在錨定 ROI 裡 */
  _targetInsideAnchor(target) {
    const a = this.anchor;
    if (!a) return false;
    const m = 2;
    return target.x >= a.x - m && target.y >= a.y - m &&
           target.x + target.w <= a.x + a.w + m &&
           target.y + target.h <= a.y + a.h + m;
  }

  /** 影像座標 → ROI 座標 */
  _toRoi(box) {
    const a = this.anchor, s = this.roiScale;
    return {
      x: (box.x - a.x) * s,
      y: (box.y - a.y) * s,
      w: box.w * s,
      h: box.h * s,
    };
  }

  /** 以錨定 ROI 裁切原生解析度影像 → 灰階 cv.Mat */
  _captureGray(source) {
    const a = this.anchor;
    if (!a || a.w < 8 || a.h < 8) return null;
    this.canvas.width = this.roiW;
    this.canvas.height = this.roiH;
    this.ctx.drawImage(source, a.x, a.y, a.w, a.h, 0, 0, this.roiW, this.roiH);
    const src = cv.imread(this.canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    src.delete();
    return gray;
  }

  // ---------- 特徵點 ----------

  _sampleInMask(gray, mask, nMax) {
    const corners = new cv.Mat();
    try {
      cv.goodFeaturesToTrack(gray, corners, nMax, 0.01, 6, mask, 3, false, 0.04);
    } catch (e) {
      corners.delete();
      return null;
    }
    if (corners.rows === 0) { corners.delete(); return null; }
    const out = new Float32Array(corners.rows * 2);
    for (let i = 0; i < corners.rows; i++) {
      out[i * 2] = corners.data32F[i * 2];
      out[i * 2 + 1] = corners.data32F[i * 2 + 1];
    }
    corners.delete();
    return out;
  }

  _rectMat(gray, box, fill) {
    const mask = cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8UC1);
    if (!fill) mask.setTo(new cv.Scalar(255));
    const x = Math.max(0, Math.round(box.x));
    const y = Math.max(0, Math.round(box.y));
    const w = Math.min(gray.cols - x, Math.round(box.w));
    const h = Math.min(gray.rows - y, Math.round(box.h));
    if (w > 0 && h > 0) {
      const roi = mask.roi(new cv.Rect(x, y, w, h));
      roi.setTo(new cv.Scalar(fill ? 255 : 0));
      roi.delete();
    }
    return mask;
  }

  /** 在 ROI 內重撒前景（目標內縮）與背景（外圈環）特徵點 */
  _resample(gray, targetRoi, ts) {
    const bag = new MatBag();
    try {
      const sh = this.cfg.flow.fgShrink;
      const inner = {
        x: targetRoi.x + targetRoi.w * sh,
        y: targetRoi.y + targetRoi.h * sh,
        w: targetRoi.w * (1 - 2 * sh),
        h: targetRoi.h * (1 - 2 * sh),
      };
      const fgMask = bag.add(this._rectMat(gray, inner, true));
      this.fgPts = this._sampleInMask(gray, fgMask, this.cfg.flow.maxFgPoints);

      // 背景環：把目標框稍微放大後排除，避免車體邊緣混進背景
      const excl = {
        x: targetRoi.x - targetRoi.w * 0.08,
        y: targetRoi.y - targetRoi.h * 0.08,
        w: targetRoi.w * 1.16,
        h: targetRoi.h * 1.16,
      };
      const bgMask = bag.add(this._rectMat(gray, excl, false));
      // 去掉畫面最外緣（常有鏡頭黑邊 / 插值瑕疵）
      const b = 4;
      cv.rectangle(bgMask, new cv.Point(0, 0), new cv.Point(gray.cols - 1, b), new cv.Scalar(0), -1);
      cv.rectangle(bgMask, new cv.Point(0, gray.rows - 1 - b), new cv.Point(gray.cols - 1, gray.rows - 1), new cv.Scalar(0), -1);
      cv.rectangle(bgMask, new cv.Point(0, 0), new cv.Point(b, gray.rows - 1), new cv.Scalar(0), -1);
      cv.rectangle(bgMask, new cv.Point(gray.cols - 1 - b, 0), new cv.Point(gray.cols - 1, gray.rows - 1), new cv.Scalar(0), -1);
      this.bgPts = this._sampleInMask(gray, bgMask, this.cfg.flow.maxBgPoints);

      this.lastSampleTs = ts;
      this._resetRef(ts);
    } finally {
      bag.freeAll();
    }
  }

  /** 把「現在的位置」設為新的量測基線起點 */
  _resetRef(ts) {
    this.fgRef = this.fgPts ? Float32Array.from(this.fgPts) : null;
    this.bgRef = this.bgPts ? Float32Array.from(this.bgPts) : null;
    this.refTs = ts;
  }

  /** 依 LK 存活點的索引，同步篩選基線參考位置 */
  static _pick(src, idx) {
    const out = new Float32Array(idx.length * 2);
    for (let k = 0; k < idx.length; k++) {
      out[k * 2] = src[idx[k] * 2];
      out[k * 2 + 1] = src[idx[k] * 2 + 1];
    }
    return out;
  }

  /**
   * LK 追蹤 + forward-backward 一致性檢核
   * @returns {prev: Float32Array, next: Float32Array} 只含通過檢核的點
   */
  _trackLK(prevGray, nextGray, ptsArr, bag) {
    if (!ptsArr || ptsArr.length < 4) return null;
    const n = ptsArr.length / 2;
    const f = this.cfg.flow;
    const win = new cv.Size(f.winSize, f.winSize);
    const crit = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 20, 0.03);

    const p0 = bag.add(cv.matFromArray(n, 1, cv.CV_32FC2, Array.from(ptsArr)));
    const p1 = bag.add(new cv.Mat());
    const st1 = bag.add(new cv.Mat());
    const er1 = bag.add(new cv.Mat());
    try {
      cv.calcOpticalFlowPyrLK(prevGray, nextGray, p0, p1, st1, er1, win, f.pyrLevels, crit);
    } catch (e) { return null; }

    // 反向追蹤：next -> prev
    const p2 = bag.add(new cv.Mat());
    const st2 = bag.add(new cv.Mat());
    const er2 = bag.add(new cv.Mat());
    try {
      cv.calcOpticalFlowPyrLK(nextGray, prevGray, p1, p2, st2, er2, win, f.pyrLevels, crit);
    } catch (e) { return null; }

    const prev = [], next = [], idx = [];
    const fbMax = f.fbErrorPx;
    for (let i = 0; i < n; i++) {
      if (st1.data[i] !== 1 || st2.data[i] !== 1) continue;
      const px = p0.data32F[i * 2], py = p0.data32F[i * 2 + 1];
      const nx = p1.data32F[i * 2], ny = p1.data32F[i * 2 + 1];
      const bx = p2.data32F[i * 2], by = p2.data32F[i * 2 + 1];
      if (nx < 0 || ny < 0 || nx >= this.roiW || ny >= this.roiH) continue;
      // forward-backward 誤差：追出去再追回來，回不到原點就是追丟了
      if (Math.hypot(bx - px, by - py) > fbMax) continue;
      prev.push(px, py); next.push(nx, ny); idx.push(i);
    }
    if (prev.length < 6) return null;
    return { prev: Float32Array.from(prev), next: Float32Array.from(next), idx };
  }

  _fitSimilarity(prev, next) {
    return fitSimilarity(prev, next, this.cfg.flow);
  }

  /**
   * 每幀呼叫一次。
   * @param source 可繪製的影像來源（video / ImageBitmap / canvas）
   * @param target 目標框（影像座標，來自 KF 預測到「現在」的框）
   * @param vw,vh  影像尺寸
   * @param ts     現在時刻（performance.now）
   * @returns 量測結果或 { ok:false, reason }
   */
  measure(source, target, vw, vh, ts) {
    if (!this.cvReady) return { ok: false, reason: 'cv-not-ready' };
    if (!target || target.w < 24 || target.h < 24) {
      return { ok: false, reason: 'target-too-small' };
    }

    const f = this.cfg.flow;
    const bag = new MatBag();
    let gray = null;

    try {
      // ---- 是否需要（重新）錨定 ----
      const needAnchor =
        !this.anchor ||
        !this.prevGray ||
        !this._targetInsideAnchor(target) ||
        !this.fgPts || !this.fgRef || !this.bgPts || !this.bgRef ||
        this.fgPts.length / 2 < f.minFgPoints ||
        (ts - this.lastSampleTs) > f.resampleMs;

      if (needAnchor) {
        this.anchor = this._computeAnchor(target, vw, vh);
        if (this.anchor.w < 24 || this.anchor.h < 24) {
          this.reset();
          return { ok: false, reason: 'anchor-too-small' };
        }
        // 原生解析度優先；只有超過上限才縮小
        const maxSide = this.cfg.camera.roiMaxSide;
        this.roiScale = Math.min(1, maxSide / Math.max(this.anchor.w, this.anchor.h));
        this.roiW = Math.max(16, Math.round(this.anchor.w * this.roiScale));
        this.roiH = Math.max(16, Math.round(this.anchor.h * this.roiScale));

        gray = this._captureGray(source);
        if (!gray) { this.reset(); return { ok: false, reason: 'capture-fail' }; }
        this._resample(gray, this._toRoi(target), ts);
        this.targetAtSample = { ...target };
        if (this.prevGray && !this.prevGray.isDeleted()) this.prevGray.delete();
        this.prevGray = gray;
        gray = null;
        this.lastTs = ts;
        return { ok: false, reason: 'reanchor', reanchored: true };
      }

      const dt = (ts - this.lastTs) / 1000;
      if (!(dt > 1e-4)) return { ok: false, reason: 'dt-zero' };

      gray = this._captureGray(source);
      if (!gray) return { ok: false, reason: 'capture-fail' };

      const targetRoi = this._toRoi(target);

      // ---- 前景點：先剔除跑出目標框的（v6 缺這一步，追丟的點會繼續投票）----
      // 基線參考位置 fgRef 必須跟著一起篩，否則兩個陣列的第 i 個點不再是同一個點。
      const keptFg = [], keptFgRef = [];
      const pad = 3;
      for (let i = 0; i < this.fgPts.length / 2; i++) {
        const x = this.fgPts[i * 2], y = this.fgPts[i * 2 + 1];
        if (x < targetRoi.x - pad || y < targetRoi.y - pad ||
            x > targetRoi.x + targetRoi.w + pad || y > targetRoi.y + targetRoi.h + pad) continue;
        keptFg.push(x, y);
        keptFgRef.push(this.fgRef[i * 2], this.fgRef[i * 2 + 1]);
      }
      const fgIn = Float32Array.from(keptFg);
      const fgRefIn = Float32Array.from(keptFgRef);
      const bgRefIn = this.bgRef;

      const fgTrack = this._trackLK(this.prevGray, gray, fgIn, bag);
      const bgTrack = this._trackLK(this.prevGray, gray, this.bgPts, bag);

      // 推進 prev（無論本 tick 量測成功與否）
      const advance = () => {
        if (fgTrack) {
          this.fgPts = fgTrack.next;
          this.fgRef = OpticalFlow._pick(fgRefIn, fgTrack.idx);
        } else {
          this.fgPts = fgIn;
          this.fgRef = fgRefIn;
        }
        if (bgTrack) {
          this.bgPts = bgTrack.next;
          this.bgRef = OpticalFlow._pick(bgRefIn, bgTrack.idx);
        }
        if (this.prevGray && !this.prevGray.isDeleted()) this.prevGray.delete();
        this.prevGray = gray;
        gray = null;
        this.lastTs = ts;
      };

      if (!fgTrack || fgTrack.prev.length / 2 < f.minFgPoints) {
        advance();
        this._resetRef(ts);      // 點數已變，舊基線的統計意義沒了
        return { ok: false, reason: 'fg-too-few', dt };
      }
      if (!bgTrack || bgTrack.prev.length / 2 < f.minBgPoints) {
        advance();
        this._resetRef(ts);
        return { ok: false, reason: 'bg-too-few', dt };
      }

      advance();

      // ---- 只有累積到足夠的基線長度才產生一筆量測 ----
      // 訊號（log 尺度變化）∝ 基線長度，雜訊（LK 次像素誤差）與基線長度無關。
      // 相鄰兩幀（40ms）的尺度變化在雜訊底下，所以這裡等到 baselineMs
      // 才用 ref → cur 擬合一次，且擬完立刻把 ref 移到現在
      //（不重疊 → 相鄰量測近似獨立 → SPRT 的累積才不會重複計算同一份證據）。
      const baseDt = (ts - this.refTs) / 1000;
      if (baseDt * 1000 < f.baselineMs) {
        return { ok: false, reason: 'accumulating', dt: baseDt };
      }
      const refTs = this.refTs;

      const fgFit = this._fitSimilarity(this.fgRef, this.fgPts);
      const bgFit = this._fitSimilarity(this.bgRef, this.bgPts);
      this._resetRef(ts);

      if (!fgFit) return { ok: false, reason: 'fg-fit-fail', dt: baseDt };
      if (!bgFit) return { ok: false, reason: 'bg-fit-fail', dt: baseDt };
      if (bgFit.inlierRatio < f.minBgInlierRatio) {
        // 背景 inlier 太少 = 背景本身不一致（多半是旁車道車流佔了多數）
        return { ok: false, reason: 'bg-inconsistent', dt: baseDt, bgInlierRatio: bgFit.inlierRatio };
      }

      // ---- 相對尺度：消掉全域縮放（對焦呼吸、自車輕微前進）----
      const sRel = fgFit.s / bgFit.s;
      const sigmaRel = Math.abs(sRel) * Math.hypot(
        fgFit.sigmaS / fgFit.s,
        bgFit.sigmaS / bgFit.s
      );

      // ---- 相對垂直位移：目標質心的實際位移 − 背景變換所預測的位移 ----
      const c = fgFit.centroid;
      const predByBg = bgFit.apply(c.x, c.y);
      const actual = fgFit.apply(c.x, c.y);
      const dyRel = (actual.y - predByBg.y) / this.roiScale;   // 換回影像 px
      const dxRel = (actual.x - predByBg.x) / this.roiScale;

      return {
        ok: true,
        dt: baseDt,
        // 這筆量測涵蓋的時間區間 —— IMU 交叉檢核必須用同一段區間積分陀螺儀，
        // 否則旋轉角與背景位移對不起來，回歸學到的增益會被系統性拉偏。
        t0: refTs,
        t1: ts,
        sRel,
        sigmaRel,
        logSRel: Math.log(sRel),
        dxRel, dyRel,
        fg: { n: fgFit.n, nIn: fgFit.nIn, s: fgFit.s, sigmaS: fgFit.sigmaS, resid: fgFit.sigmaResid, rRms: fgFit.rRms },
        bg: { n: bgFit.n, nIn: bgFit.nIn, s: bgFit.s, inlierRatio: bgFit.inlierRatio, resid: bgFit.sigmaResid },
        // 給 IMU 校準用的背景觀測（ROI 座標）
        bgObs: { dx: bgFit.tx, dy: bgFit.ty, dtheta: bgFit.theta },
        roi: { ...this.anchor, scale: this.roiScale, w: this.roiW, h: this.roiH },
      };
    } finally {
      bag.freeAll();
      if (gray && !gray.isDeleted()) gray.delete();
    }
  }
}
