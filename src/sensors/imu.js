// =============================================
// IMU（陀螺儀 + 加速度計）
// =============================================
// 用途有三：
//   1. ego-motion：陀螺儀直接量到相機旋轉。手機小幅晃動 99% 是旋轉，
//      而轉 1° 在 1280px / 60°HFOV 的畫面上就是 ~21px —— 遠大於前車起步的
//      真實訊號（幾 px）。所以旋轉是壓倒性的雜訊來源，而陀螺儀直接量它。
//   2. 地平線：加速度計的重力向量給出相機俯角 → 畫面上的地平線位置，
//      取代寫死的 FOE_Y_RATIO = 0.45。
//   3. 自車靜止判定：怠速停車與行駛的加速度變異數特徵不同，
//      作為 GPS speed 為 null 時的即時備援。
//
// 「陀螺儀角度 → 畫面位移」的增益（含符號、焦距、安裝姿態）不寫死，
// 而是用 RLS 線上回歸「整合角度」對「視覺觀測到的背景位移」學出來。
// 這樣不需要知道手機怎麼擺，也不需要手填 FOV。

import { Rls, clamp, variance } from '../util/math.js';
import { CONFIG } from '../config.js';

const DEG2RAD = Math.PI / 180;

export class ImuSensor {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.samples = [];          // { t, rx, ry, rz (rad/s), ax, ay, az (m/s^2) }
    this.available = false;
    this.permission = 'unknown';
    this._handler = null;

    // 陀螺儀 -> 畫面 (dx, dy, dtheta) 的線上校準
    // 輸入 u = [gx, gy, gz]（本 tick 整合出的旋轉角，rad）
    this.calibX = new Rls(3);
    this.calibY = new Rls(3);
    this.calibR = new Rls(3);

    // 自車靜止用的加速度變異數基線（GPS 確認靜止時學習）
    this.stillAccelVar = null;
    this.lastAccelVar = 0;
    // 沒有 GPS 可教時的自助基線：本次行程觀察到的最安靜狀態
    this.minAccelVar = null;
  }

  async start() {
    if (typeof window === 'undefined' || !window.DeviceMotionEvent) {
      this.permission = 'unsupported';
      return false;
    }
    // iOS 13+ 需要使用者手勢下的明確授權
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const res = await DeviceMotionEvent.requestPermission();
        this.permission = res;
        if (res !== 'granted') return false;
      } catch (e) {
        this.permission = 'denied';
        return false;
      }
    } else {
      this.permission = 'granted';
    }

    this._handler = (e) => this._onMotion(e);
    window.addEventListener('devicemotion', this._handler);
    return true;
  }

  stop() {
    if (this._handler) window.removeEventListener('devicemotion', this._handler);
    this._handler = null;
    this.samples.length = 0;
    this.available = false;
  }

  _onMotion(e) {
    const t = performance.now();
    const rr = e.rotationRate;
    const ag = e.accelerationIncludingGravity || e.acceleration;
    if (!rr && !ag) return;
    this.samples.push({
      t,
      // DeviceMotion 的 rotationRate 單位是度/秒
      rx: rr && rr.beta != null ? rr.beta * DEG2RAD : 0,
      ry: rr && rr.gamma != null ? rr.gamma * DEG2RAD : 0,
      rz: rr && rr.alpha != null ? rr.alpha * DEG2RAD : 0,
      ax: ag && ag.x != null ? ag.x : 0,
      ay: ag && ag.y != null ? ag.y : 0,
      az: ag && ag.z != null ? ag.z : 0,
      hasGyro: !!(rr && (rr.alpha != null || rr.beta != null || rr.gamma != null)),
    });
    this.available = true;
    const cutoff = t - this.cfg.imu.bufferMs;
    while (this.samples.length && this.samples[0].t < cutoff) this.samples.shift();
  }

  get hasGyro() {
    const n = this.samples.length;
    return n > 2 && this.samples[n - 1].hasGyro;
  }

  /**
   * 把 [t0, t1] 區間的角速度以梯形法積分成旋轉角（rad）
   * 回傳 [gx, gy, gz]，對應裝置 x/y/z 軸。
   */
  integrateRotation(t0, t1) {
    const out = [0, 0, 0];
    const s = this.samples;
    if (s.length < 2 || !(t1 > t0)) return out;
    for (let i = 1; i < s.length; i++) {
      const a = s[i - 1], b = s[i];
      const lo = Math.max(a.t, t0), hi = Math.min(b.t, t1);
      if (hi <= lo) continue;
      const span = b.t - a.t;
      if (!(span > 0)) continue;
      const dtSec = (hi - lo) / 1000;
      // 在 [lo, hi] 兩端線性內插後取平均，再乘時距（梯形法）
      const f = (t) => (b.t - t) / span;
      const w0 = f(lo), w1 = f(hi);
      out[0] += 0.5 * ((a.rx * w0 + b.rx * (1 - w0)) + (a.rx * w1 + b.rx * (1 - w1))) * dtSec;
      out[1] += 0.5 * ((a.ry * w0 + b.ry * (1 - w0)) + (a.ry * w1 + b.ry * (1 - w1))) * dtSec;
      out[2] += 0.5 * ((a.rz * w0 + b.rz * (1 - w0)) + (a.rz * w1 + b.rz * (1 - w1))) * dtSec;
    }
    return out;
  }

  /** 用學到的增益預測本 tick 的畫面 ego-motion（ROI 座標） */
  predictImageMotion(rot) {
    return {
      dx: this.calibX.predict(rot),
      dy: this.calibY.predict(rot),
      dtheta: this.calibR.predict(rot),
      quality: Math.min(this.calibX.quality, this.calibY.quality),
      samples: this.calibX.count,
    };
  }

  /** 把視覺背景觀測到的位移餵回去，校準陀螺儀增益 */
  teachImageMotion(rot, obs) {
    if (!this.hasGyro) return;
    const mag = Math.abs(rot[0]) + Math.abs(rot[1]) + Math.abs(rot[2]);
    if (mag < 1e-4) return;         // 完全沒轉的樣本對回歸沒有資訊量
    this.calibX.update(rot, obs.dx);
    this.calibY.update(rot, obs.dy);
    this.calibR.update(rot, obs.dtheta);
  }

  get calibrated() {
    const c = this.cfg.imu;
    return this.calibX.count >= c.calibMinSamples &&
           Math.min(this.calibX.quality, this.calibY.quality) >= c.calibMinQuality;
  }

  /**
   * 地平線在畫面上的 y（0~1）。由重力向量推得相機俯角：
   *   俯角 pitch = atan2(-az, -ay)（後鏡頭光軸 ~ 裝置 -z，畫面向下 ~ 裝置 -y）
   *   地平線 y = H/2 - f*tan(pitch)
   * f 用學到的陀螺儀增益反推。資料不足時回 null，由呼叫端 fallback。
   */
  horizonRatio(focalPx, frameH) {
    const s = this.samples;
    if (s.length < 5 || !focalPx || !frameH) return null;
    const t1 = s[s.length - 1].t;
    let ax = 0, ay = 0, az = 0, n = 0;
    for (let i = s.length - 1; i >= 0 && t1 - s[i].t < 500; i--) {
      ax += s[i].ax; ay += s[i].ay; az += s[i].az; n++;
    }
    if (n < 3) return null;
    ax /= n; ay /= n; az /= n;
    const g = Math.hypot(ax, ay, az);
    if (!(g > 5 && g < 15)) return null;   // 不像純重力（正在強烈加速）-> 放棄
    const pitch = Math.atan2(-az, -ay);
    const yPx = frameH / 2 - focalPx * Math.tan(pitch);
    const r = yPx / frameH;
    return (r > -0.5 && r < 1.5) ? clamp(r, 0.05, 0.95) : null;
  }

  /** 學到的焦距（px）。以校準增益的範數估計（增益單位是 px/rad） */
  focalPxEstimate() {
    if (!this.calibrated) return null;
    const wx = this.calibX.w, wy = this.calibY.w;
    const f = Math.max(
      Math.hypot(wx[0], wx[1], wx[2]),
      Math.hypot(wy[0], wy[1], wy[2])
    );
    return f > 10 && f < 1e5 ? f : null;
  }

  /**
   * 近期加速度大小的變異數，用於自車靜止判定。
   * 同時維護一個「本次行程觀察到的最安靜狀態」——這是靜止基線的自助
   * (bootstrap) 來源：GPS 在紅燈停車時經常回傳 null，如果只靠 GPS 教基線，
   * 就會出現「等不到 GPS 教 → IMU 無基線 → 自車狀態永遠 unknown → 永不警示」
   * 的死結。最小值會以每次取樣 +2% 的速度緩慢上漂，讓它能適應環境變化。
   */
  accelVariance() {
    const s = this.samples;
    if (s.length < 8) return null;
    const t1 = s[s.length - 1].t;
    const win = this.cfg.ego.imuWindowMs;
    const mags = [];
    for (let i = s.length - 1; i >= 0 && t1 - s[i].t < win; i--) {
      mags.push(Math.hypot(s[i].ax, s[i].ay, s[i].az));
    }
    if (mags.length < 8) return null;
    const v = variance(mags);
    this.lastAccelVar = v;
    this.minAccelVar = this.minAccelVar === null
      ? v
      : Math.min(v, this.minAccelVar * 1.02 + 1e-6);
    return v;
  }

  /** GPS 確認靜止時呼叫，學習靜止基線 */
  teachStillBaseline() {
    const v = this.accelVariance();
    if (v === null) return;
    this.stillAccelVar = this.stillAccelVar === null
      ? v
      : 0.95 * this.stillAccelVar + 0.05 * v;
  }

  /** IMU 判斷是否移動：null = 無法判斷 */
  isMovingByImu() {
    const v = this.accelVariance();
    if (v === null) return null;
    // 優先用 GPS 教出來的基線；沒有就用本次行程的最安靜狀態自助
    const base = this.stillAccelVar !== null ? this.stillAccelVar : this.minAccelVar;
    if (base === null) return null;
    return v > Math.max(base, 1e-4) * this.cfg.ego.imuMoveFactor;
  }
}
