// =============================================
// 第二條量測路徑：bbox 尺度變化率
// =============================================
// 為什麼要有第二條路：
//
//   光流（optical expansion）是比較準的估計器，但它的**可用率**很低。
//   2026-09-10 那支停等車陣的實測：135 個 tick 裡只有 36 個產出量測（27%），
//   其餘是 `accumulating`（240ms 基線還沒湊滿）與 `reanchor`（目標跑出錨定框）。
//   而前車真正起步的窗口只有 1.8 秒 —— 一條 27% 可用率、中斷成本 240ms
//   的單一路徑，在這個窗口裡跑不完。實測結果是 LLR 走到 7.4/9.2 就被
//   目標換手清空。
//
//   這不是門檻問題，是「串聯單點失效」的結構問題。解法是冗餘。
//
// 為什麼選 bbox 尺度：
//
//   這是量產 camera-only FCW 的**基礎作法**（Mobileye 2004 就發表了
//   用尺寸變化算 TTC）。A-Eye 目前只裝了進階版（對前景/背景分別擬合
//   相似變換取尺度比 = optical expansion，Yang & Ramanan CVPR'20），
//   缺的反而是這個基礎版。而且它已經被算出來了 —— `KalmanBox.logScaleRate`
//   一直存在，只用在自車結構黑名單，從來沒餵給起步判定。
//
//   它的失效模式與光流**互補**：
//     光流需要紋理（夜間車尾常常撒不出點）、需要 240ms 基線、需要背景環
//     bbox 只需要偵測器看得到車
//
// 三個必須守住的紀律：
//
//   1. **不能餵 KF 的濾波後狀態。** `kf.lw.v` 是被平滑過的速度，相鄰時刻
//      高度相關 —— 把它每個 tick 丟進 SPRT，就是 v6 「把相關觀測當獨立」
//      那個錯誤的第二次。這裡只吃**原始偵測框**（`track.lastBox`）。
//
//   2. **不重疊的固定基線。** 與光流的 `baselineMs` 完全同一套規矩：
//      累積一個窗口 → 擬合 → 送出一筆量測 → 清空重來。
//      相鄰量測不共用任何一筆觀測，SPRT 的獨立性假設才成立。
//
//   3. **σ 由資料自己估。** 每個窗口的殘差散布用 EMA 累積成「每筆偵測的
//      尺寸雜訊」，斜率的標準誤由最小平方傳播得到。config 裡的值只當下限
//      （3 個點的殘差估計本身太不可靠，需要一個地板）。
//
// 已知限制（刻意不補償，因為上游已經有閘門）：
//   光流量的是 s_fg/s_bg —— 相對背景，自車前進造成的全域縮放被除掉了。
//   bbox 沒有這個除法：自車往前開時，前車的框也會變大。
//   所以這條路**只在自車靜止時有效** —— 而 `departure.update()` 的
//   `ctx.egoStill` 本來就是硬前提，不需要在這裡再做一次。

import { CONFIG } from '../config.js';

export class BboxScaleEstimator {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.reset();
  }

  reset() {
    this.trackId = null;
    this.samples = [];        // { t(秒), ls(log 尺寸), cy }
    this.windowStart = 0;
    // 每筆偵測的 log 尺寸雜訊，由殘差線上學習。初值取設定的下限。
    this.sigmaPt = this.cfg.bboxScale.sigmaFloor;
    this.lastReason = 'idle';
    this.emitted = 0;
  }

  /**
   * 每個 tick 呼叫一次。
   * @param track 目前的前車 track（需要 lastBox / lastBoxTs）
   * @param now   現在時刻（ms）
   * @returns 與 OpticalFlow.measure() 同形狀的量測，或 { ok:false, reason }
   */
  update(track, now) {
    const c = this.cfg.bboxScale;
    if (!track || !track.lastBox) { this.lastReason = 'no-target'; return { ok: false, reason: 'no-target' }; }

    // 換了 track 就重來 —— 兩台不同車的尺寸序列接在一起是沒有意義的
    if (track.id !== this.trackId) {
      this.trackId = track.id;
      this.samples.length = 0;
      this.windowStart = 0;
    }

    // ---- 只吃「新的原始量測」----
    const ts = track.lastBoxTs;
    const b = track.lastBox;
    const last = this.samples[this.samples.length - 1];
    if (!last || ts > last.tMs) {
      if (b.w > 1 && b.h > 1) {
        this.samples.push({
          tMs: ts,
          t: ts / 1000,
          ls: (Math.log(b.w) + Math.log(b.h)) / 2,
          cy: b.y + b.h / 2,
        });
        if (!this.windowStart) this.windowStart = ts;
      }
    }

    const span = this.samples.length ? this.samples[this.samples.length - 1].tMs - this.windowStart : 0;
    if (this.samples.length < c.minSamples || span < c.baselineMs) {
      this.lastReason = 'accumulating';
      return { ok: false, reason: 'accumulating', n: this.samples.length, spanMs: span };
    }

    const m = this._fit();
    // 不重疊：擬完就清空，下一個窗口從零開始（見檔頭紀律 2）
    this.samples.length = 0;
    this.windowStart = 0;
    if (m.ok) this.emitted++;
    this.lastReason = m.ok ? 'ok' : m.reason;
    return m;
  }

  /** 對窗口內的 (t, logSize) 與 (t, cy) 各做一次最小平方直線擬合 */
  _fit() {
    const c = this.cfg.bboxScale;
    const S = this.samples;
    const n = S.length;
    const mt = S.reduce((a, s) => a + s.t, 0) / n;
    const mls = S.reduce((a, s) => a + s.ls, 0) / n;
    const mcy = S.reduce((a, s) => a + s.cy, 0) / n;

    let stt = 0, sls = 0, scy = 0;
    for (const s of S) {
      const dt = s.t - mt;
      stt += dt * dt;
      sls += dt * (s.ls - mls);
      scy += dt * (s.cy - mcy);
    }
    if (!(stt > 1e-9)) return { ok: false, reason: 'degenerate' };

    const kLs = sls / stt;              // d log(尺寸)/dt  →  V = −kLs
    const kCy = scy / stt;              // 中心 y 的速度（px/s）

    // ---- 殘差 → 每筆偵測的尺寸雜訊（線上學習）----
    let ss = 0;
    for (const s of S) {
      const r = (s.ls - mls) - kLs * (s.t - mt);
      ss += r * r;
    }
    const dof = Math.max(n - 2, 1);
    const sigmaWin = Math.sqrt(ss / dof);
    // 單一窗口只有 1~3 個自由度，估計本身很不穩 → 用 EMA 累積成長期值，
    // 並以 config 的值當地板（不是當真值）
    this.sigmaPt = (1 - c.sigmaEma) * this.sigmaPt + c.sigmaEma * sigmaWin;
    // 地板的用途只是「避免某個窗口剛好殘差為 0 就宣稱無限精確」，
    // 不是拿來當雜訊的真值 —— 設太高會直接把估計器的精度上限鎖死。
    // 1% 的依據：框座標的定位精度約 1px，而近距離前車的框寬 250~300px
    //（→ 0.35%），取 1% 已有 3 倍餘裕。真實雜訊由 EMA 自己學。
    const sigmaPt = Math.max(this.sigmaPt, c.sigmaFloor);

    // 最小平方的斜率標準誤：σ_k = σ_點 / sqrt(Σ(t−t̄)²)
    const sigmaK = sigmaPt / Math.sqrt(stt);

    const dt = S[n - 1].t - S[0].t;
    if (!(dt > 1e-3)) return { ok: false, reason: 'dt-zero' };

    const logSRel = kLs * dt;
    const sRel = Math.exp(logSRel);
    const sigmaRel = Math.abs(sRel) * sigmaK * dt;
    if (!(sigmaRel > 0) || !isFinite(sigmaRel)) return { ok: false, reason: 'sigma-bad' };

    return {
      ok: true,
      source: 'bbox',
      dt,
      t0: S[0].tMs,
      t1: S[n - 1].tMs,
      sRel,
      sigmaRel,
      logSRel,
      // 「往地平線方向移動」的佐證。注意這裡**沒有**扣掉背景運動
      //（光流那條路是扣過的），所以它會吃到手機的俯仰晃動。
      // 自車靜止時晃動在窗口尺度上大致對消，而它只是佐證、不是主判據。
      dxRel: 0,
      dyRel: kCy * dt,
      n,
      sigmaPt,
      fg: { n, nIn: n, s: sRel, sigmaS: sigmaRel },
      bg: { n: 0, nIn: 0, s: 1, sigmaS: 0, inlierRatio: 1 },
    };
  }

  debugLine() {
    return `bbox ${this.lastReason} n=${this.samples.length}`
      + ` σpt=${this.sigmaPt.toFixed(3)} emitted=${this.emitted}`;
  }
}
