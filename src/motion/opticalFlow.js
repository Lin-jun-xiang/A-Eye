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

import { madSigma } from '../util/math.js';
import { CONFIG } from '../config.js';

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
    this.fgPts = null;          // [x0,y0,x1,y1,...] ROI 座標
    this.bgPts = null;
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
      try {
        await loadScript('https://docs.opencv.org/4.9.0/opencv.js');
      } catch (e) {
        console.warn('[A-Eye] OpenCV.js 載入失敗:', e.message);
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
    } finally {
      bag.freeAll();
    }
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

    const prev = [], next = [];
    const fbMax = f.fbErrorPx;
    for (let i = 0; i < n; i++) {
      if (st1.data[i] !== 1 || st2.data[i] !== 1) continue;
      const px = p0.data32F[i * 2], py = p0.data32F[i * 2 + 1];
      const nx = p1.data32F[i * 2], ny = p1.data32F[i * 2 + 1];
      const bx = p2.data32F[i * 2], by = p2.data32F[i * 2 + 1];
      if (nx < 0 || ny < 0 || nx >= this.roiW || ny >= this.roiH) continue;
      // forward-backward 誤差：追出去再追回來，回不到原點就是追丟了
      if (Math.hypot(bx - px, by - py) > fbMax) continue;
      prev.push(px, py); next.push(nx, ny);
    }
    if (prev.length < 6) return null;
    return { prev: Float32Array.from(prev), next: Float32Array.from(next) };
  }

  /**
   * 擬合相似變換（4 自由度：平移 + 旋轉 + 尺度），並回報估計的不確定度。
   * σ_s 的傳播：尺度是「以質心為原點的徑向縮放」，
   *   σ_s ≈ σ_residual / (rRms · √n)
   * 這是最小平方估計的標準誤，不是憑感覺的門檻。
   */
  _fitSimilarity(prev, next, bag) {
    const n = prev.length / 2;
    if (n < 4) return null;
    const f = this.cfg.flow;
    const from = bag.add(cv.matFromArray(n, 1, cv.CV_32FC2, Array.from(prev)));
    const to = bag.add(cv.matFromArray(n, 1, cv.CV_32FC2, Array.from(next)));
    const inl = bag.add(new cv.Mat());
    let M = null;
    try {
      M = cv.estimateAffinePartial2D(from, to, inl, cv.RANSAC, f.ransacReprojPx, 2000, 0.99, 10);
    } catch (e) { return null; }
    if (!M || M.empty() || M.rows !== 2 || M.cols !== 3) { if (M) M.delete(); return null; }
    bag.add(M);

    const a = M.doubleAt(0, 0), b = M.doubleAt(0, 1);
    const tx = M.doubleAt(0, 2), ty = M.doubleAt(1, 2);
    const s = Math.hypot(a, b);
    const theta = Math.atan2(-b, a);
    if (!(s > 0.5 && s < 2.0)) return null;    // 離譜的解，視為擬合失敗

    // inlier 上的殘差與半徑
    const resid = [];
    let sumX = 0, sumY = 0, nIn = 0;
    const hasInl = inl && inl.rows === n;
    for (let i = 0; i < n; i++) {
      if (hasInl && inl.data[i] !== 1) continue;
      sumX += prev[i * 2]; sumY += prev[i * 2 + 1];
      nIn++;
    }
    if (nIn < 4) return null;
    const cx = sumX / nIn, cy = sumY / nIn;
    let sumR2 = 0;
    for (let i = 0; i < n; i++) {
      if (hasInl && inl.data[i] !== 1) continue;
      const px = prev[i * 2], py = prev[i * 2 + 1];
      const ex = a * px + b * py + tx;
      const ey = -b * px + a * py + ty;
      resid.push(Math.hypot(next[i * 2] - ex, next[i * 2 + 1] - ey));
      sumR2 += (px - cx) * (px - cx) + (py - cy) * (py - cy);
    }
    const rRms = Math.sqrt(sumR2 / nIn);
    const sigmaResid = Math.max(madSigma(resid), 0.05);   // 下限防止除以 0
    const sigmaS = rRms > 1e-3
      ? sigmaResid / (rRms * Math.sqrt(nIn))
      : Infinity;

    return {
      a, b, tx, ty, s, theta,
      sigmaS, sigmaResid, rRms,
      n, nIn, inlierRatio: nIn / n,
      centroid: { x: cx, y: cy },
      apply: (x, y) => ({ x: a * x + b * y + tx, y: -b * x + a * y + ty }),
    };
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
        !this.fgPts || this.fgPts.length / 2 < f.minFgPoints ||
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
      const keptFg = [];
      const pad = 3;
      for (let i = 0; i < this.fgPts.length / 2; i++) {
        const x = this.fgPts[i * 2], y = this.fgPts[i * 2 + 1];
        if (x < targetRoi.x - pad || y < targetRoi.y - pad ||
            x > targetRoi.x + targetRoi.w + pad || y > targetRoi.y + targetRoi.h + pad) continue;
        keptFg.push(x, y);
      }
      const fgIn = Float32Array.from(keptFg);

      const fgTrack = this._trackLK(this.prevGray, gray, fgIn, bag);
      const bgTrack = this._trackLK(this.prevGray, gray, this.bgPts, bag);

      // 推進 prev（無論本 tick 量測成功與否）
      const advance = () => {
        if (fgTrack) this.fgPts = fgTrack.next;
        if (bgTrack) this.bgPts = bgTrack.next;
        if (this.prevGray && !this.prevGray.isDeleted()) this.prevGray.delete();
        this.prevGray = gray;
        gray = null;
        this.lastTs = ts;
      };

      if (!fgTrack || fgTrack.prev.length / 2 < f.minFgPoints) {
        advance();
        return { ok: false, reason: 'fg-too-few', dt };
      }
      if (!bgTrack || bgTrack.prev.length / 2 < f.minBgPoints) {
        advance();
        return { ok: false, reason: 'bg-too-few', dt };
      }

      const fgFit = this._fitSimilarity(fgTrack.prev, fgTrack.next, bag);
      const bgFit = this._fitSimilarity(bgTrack.prev, bgTrack.next, bag);
      advance();

      if (!fgFit) return { ok: false, reason: 'fg-fit-fail', dt };
      if (!bgFit) return { ok: false, reason: 'bg-fit-fail', dt };
      if (bgFit.inlierRatio < f.minBgInlierRatio) {
        // 背景 inlier 太少 = 背景本身不一致（多半是旁車道車流佔了多數）
        return { ok: false, reason: 'bg-inconsistent', dt, bgInlierRatio: bgFit.inlierRatio };
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
        dt,
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
