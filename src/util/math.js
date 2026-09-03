// =============================================
// 通用數學 / 統計工具
// =============================================

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export function median(arr) {
  if (!arr || arr.length === 0) return 0;
  const s = Float64Array.from(arr).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** 中位數絕對偏差 → 標準差的穩健估計（常數 1.4826 讓它在常態下無偏） */
export function madSigma(arr) {
  if (!arr || arr.length === 0) return 0;
  const med = median(arr);
  const dev = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) dev[i] = Math.abs(arr[i] - med);
  return 1.4826 * median(dev);
}

export function mean(arr) {
  if (!arr || arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

export function variance(arr) {
  if (!arr || arr.length < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) { const d = arr[i] - m; s += d * d; }
  return s / (arr.length - 1);
}

/**
 * 標準常態分佈的反累積分佈函數（Acklam 演算法，精度 ~1e-9）
 * 用途：把「可容忍誤警率 α」直接換成「需要幾個標準差 z」，
 *       這樣 config 裡寫的是有物理意義的機率，而不是拍腦袋的 z 值。
 */
export function normInv(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
  const pLow = 0.02425, pHigh = 1 - pLow;
  let q, r;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---------- bbox 工具（統一用 {x, y, w, h}）----------

export function iou(a, b) {
  if (!a || !b) return 0;
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = x2 - x1, ih = y2 - y1;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

export const boxCenter = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });
export const boxBottom = (b) => b.y + b.h;

export function boxFromArray(arr) {
  return { x: arr[0], y: arr[1], w: arr[2], h: arr[3] };
}

export function inflateBox(b, factor, maxW, maxH) {
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  const w = b.w * factor, h = b.h * factor;
  let x = cx - w / 2, y = cy - h / 2;
  let ww = w, hh = h;
  if (x < 0) { ww += x; x = 0; }
  if (y < 0) { hh += y; y = 0; }
  if (x + ww > maxW) ww = maxW - x;
  if (y + hh > maxH) hh = maxH - y;
  return { x, y, w: ww, h: hh };
}

/**
 * 遞迴最小平方（RLS）單輸出線性回歸：y ≈ wᵀ·u
 * 用來線上學習「陀螺儀角度 → 畫面位移」的增益（含符號與焦距），
 * 不需要事先知道手機安裝姿態，也不需要手填 FOV。
 */
export class Rls {
  constructor(dim, { lambda = 0.995, delta = 1e3 } = {}) {
    this.n = dim;
    this.lambda = lambda;
    this.w = new Float64Array(dim);
    this.P = new Float64Array(dim * dim);
    for (let i = 0; i < dim; i++) this.P[i * dim + i] = delta;
    this.count = 0;
    this._resVar = 0;   // 殘差變異數（EWMA）
    this._sigVar = 0;   // 訊號變異數（EWMA）
  }
  predict(u) {
    let y = 0;
    for (let i = 0; i < this.n; i++) y += this.w[i] * u[i];
    return y;
  }
  update(u, y) {
    const n = this.n, P = this.P, lam = this.lambda;
    // Pu = P·u
    const Pu = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += P[i * n + j] * u[j];
      Pu[i] = s;
    }
    let uPu = 0;
    for (let i = 0; i < n; i++) uPu += u[i] * Pu[i];
    const denom = lam + uPu;
    if (!(denom > 1e-12)) return;
    const err = y - this.predict(u);
    for (let i = 0; i < n; i++) this.w[i] += (Pu[i] / denom) * err;
    // P = (P − Pu·Puᵀ/denom) / lam
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        P[i * n + j] = (P[i * n + j] - (Pu[i] * Pu[j]) / denom) / lam;
    this.count++;
    const a = 0.02;
    this._resVar = (1 - a) * this._resVar + a * err * err;
    this._sigVar = (1 - a) * this._sigVar + a * y * y;
  }
  /** 殘差的標準差估計（供「預測值與觀測值是否顯著不符」的檢定用） */
  get residSigma() { return Math.sqrt(Math.max(this._resVar, 0)); }

  /** 擬合品質 0~1（近似 R²）；樣本不足時回 0 */
  get quality() {
    if (this.count < 40 || this._sigVar < 1e-9) return 0;
    return clamp(1 - this._resVar / this._sigVar, 0, 1);
  }
}

/**
 * 一維等速卡爾曼濾波（狀態 = [位置, 速度]）
 * 支援可變 dt——這是取代 EMA 的關鍵：EMA 假設固定取樣率，
 * 幀率一抖動就產生相位滯後；KF 用 dt 明確建模。
 */
export class Kf1d {
  /** @param q 過程雜訊（速度的隨機遊走強度，單位 (值/秒²)²/秒 */
  constructor({ q = 1, r = 1 } = {}) {
    this.q = q; this.r = r;
    this.x = 0; this.v = 0;
    this.P = [[1e6, 0], [0, 1e6]];
    this.init = false;
  }
  reset() { this.init = false; this.x = 0; this.v = 0; this.P = [[1e6, 0], [0, 1e6]]; }
  predict(dt) {
    if (!this.init || !(dt > 0)) return;
    const { P, q } = this;
    this.x += this.v * dt;
    // F = [[1, dt], [0, 1]]
    const p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1];
    const p01 = P[0][1] + dt * P[1][1];
    const p10 = P[1][0] + dt * P[1][1];
    const p11 = P[1][1];
    // 連續時間等速模型的過程雜訊（標準形式）
    const dt2 = dt * dt, dt3 = dt2 * dt;
    P[0][0] = p00 + q * dt3 / 3;
    P[0][1] = p01 + q * dt2 / 2;
    P[1][0] = p10 + q * dt2 / 2;
    P[1][1] = p11 + q * dt;
  }
  /** 觀測位置 z，觀測變異數 r（可逐次給，變異數加權） */
  update(z, r = this.r) {
    if (!this.init) {
      this.x = z; this.v = 0; this.init = true;
      this.P = [[r, 0], [0, 1e4]];
      return;
    }
    const P = this.P;
    const S = P[0][0] + r;
    if (!(S > 1e-12)) return;
    const k0 = P[0][0] / S, k1 = P[1][0] / S;
    const y = z - this.x;
    this.x += k0 * y;
    this.v += k1 * y;
    const p00 = P[0][0], p01 = P[0][1], p10 = P[1][0], p11 = P[1][1];
    P[0][0] = p00 - k0 * p00;
    P[0][1] = p01 - k0 * p01;
    P[1][0] = p10 - k1 * p00;
    P[1][1] = p11 - k1 * p01;
  }
  get posVar() { return this.P[0][0]; }
  get velVar() { return this.P[1][1]; }
}
