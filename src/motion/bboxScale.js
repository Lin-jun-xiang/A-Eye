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
//   4. **尺度只用寬度量，而且被畫面裁切的邊不算量測。** 見下。
//
// ---- 紀律 4 的由來：手持 + 儀表板入鏡的實測（2026-09-11）----
//
// 原本的尺度是 √(w·h)。在手機架在擋風玻璃、畫面裡沒有內裝時這沒問題，
// 但手持時儀表板會遮住前車的下半 —— 於是偵測器的框**底邊在「車尾可見底部」
// 與「車尾＋儀表板」之間反覆跳**，高度變成一個沒有意義的量。
//
// 同一台車、同一段時間的實測（720x1280 手持夜景，前車在 t≈2.2s 才起步）：
//
//   車靜止的 0.2~2.0s     寬度 581 582 587 584 590 583 584 589 589   變異 0.78%
//                        高度 370 378 645 624 627 631 629 638 439   變異 22.6%
//   起步的 2.0~4.8s       寬度 574 → 217（單調），推得 V = 0.347（TTC 2.9 秒）
//
//   一個 600ms 窗口內的訊號約 19%，於是訊噪比：
//       寬度 24 : 1        高度 1 : 1        √(w·h) 約 1.6 : 1
//
// 高度**完全不帶資訊**，而 √(w·h) 等於拿乾淨訊號去和純雜訊平均。
// 實測後果：車明明停著，√(w·h) 卻量出 V = 0.251（中位）、最高 0.507，
// LLR 衝到 11.6（門檻 9.2）—— 這是誤報，不是靈敏度問題。
// 改用寬度之後靜止期的雜訊降到 0.78%。
//
// 順帶一提，這也和專案自己的地面真值一致：README 裡那條 LED 尾燈帶的
// 真值量的就是**寬度**，而不是面積。
//
// **被裁切的邊不算量測**（censored，不是 outlier）：
//   自車儀表板被偵測成 car 時，它的框左右兩緣都貼著畫面邊
//   （實測 5/5 幀，x=0 且 x+w=W），於是寬度被畫面卡死、變異恆為 0.0%。
//   那**不是**「尺度穩定」的證據，而是「這個量根本沒被量到」。
//   若把它當成有效量測，任何偵測抖動都會被當成真實的尺度變化 ——
//   實測內裝框量出 V = 0.061（是 minInvTtc 的 3 倍、TTC 16 秒），
//   物理上就長得像「一台正在緩慢駛離的車」。
//   所以規則是拒絕量測，而不是假設抖動夠小：
//   **左右兩緣同時被畫面裁切 → 不產出量測。**
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
   * @param vw    影像寬（用來判斷框的左右緣有沒有被畫面裁掉，見紀律 4）
   * @param vh    影像高（目前只為對稱保留，尺度不使用高度）
   * @returns 與 OpticalFlow.measure() 同形狀的量測，或 { ok:false, reason }
   */
  update(track, now, vw, vh) {
    const c = this.cfg.bboxScale;
    if (!track || !track.lastBox) { this.lastReason = 'no-target'; return { ok: false, reason: 'no-target' }; }

    // 換了 track 就重來 —— 兩台不同車的尺寸序列接在一起是沒有意義的
    if (track.id !== this.trackId) {
      this.trackId = track.id;
      this.samples.length = 0;
      this.windowStart = 0;
    }

    // ---- 被畫面裁切的寬度是「設限」而不是量測（紀律 4）----
    // 左右兩緣同時貼著畫面邊 → 寬度被畫面卡死，不管物體實際多寬都量到 W。
    // 這種框不可能產出有意義的尺度序列，直接拒絕，並清掉已累積的樣本
    //（半個窗口是真寬度、半個窗口是被卡住的 W，擬出來的斜率是假的）。
    const b = track.lastBox;
    if (Number.isFinite(vw)) {
      const clip = c.edgeClipPx;
      if (b.x <= clip && b.x + b.w >= vw - clip) {
        this.samples.length = 0;
        this.windowStart = 0;
        this.lastReason = 'width-censored';
        return { ok: false, reason: 'width-censored' };
      }
    }

    // ---- 只吃「新的原始量測」----
    const ts = track.lastBoxTs;
    const last = this.samples[this.samples.length - 1];
    if (!last || ts > last.tMs) {
      if (b.w > 1 && b.h > 1) {
        this.samples.push({
          tMs: ts,
          t: ts / 1000,
          // 尺度 = 寬度。不用 √(w·h) —— 底邊被自車內裝遮住時高度是純雜訊
          // （實測訊噪比 寬度 24:1、高度 1:1），見檔頭紀律 4。
          ls: Math.log(b.w),
          // 位置佐證也改用**上緣**而不是中心：中心 = y + h/2 同樣吃到被污染的
          // 高度（實測車靜止時中心擺盪 130px，而上緣只有 11px）。
          top: b.y,
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

  /** 對窗口內的 (t, log寬度) 與 (t, 上緣y) 各做一次最小平方直線擬合 */
  _fit() {
    const c = this.cfg.bboxScale;
    const S = this.samples;
    const n = S.length;
    const mt = S.reduce((a, s) => a + s.t, 0) / n;
    const mls = S.reduce((a, s) => a + s.ls, 0) / n;
    const mcy = S.reduce((a, s) => a + s.top, 0) / n;

    let stt = 0, sls = 0, scy = 0;
    for (const s of S) {
      const dt = s.t - mt;
      stt += dt * dt;
      sls += dt * (s.ls - mls);
      scy += dt * (s.top - mcy);
    }
    if (!(stt > 1e-9)) return { ok: false, reason: 'degenerate' };

    const kLs = sls / stt;              // d log(寬度)/dt  →  V = −kLs
    const kTop = scy / stt;             // **上緣** y 的速度（px/s）

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
      //
      // 取負號的理由：下游的約定是 `dyRel < 0` 代表「往地平線方向移動」，
      // 而這裡量的是**上緣**。車遠離時整個框往地平線收斂 ——
      // 底邊往上、**上緣往下**（實測 #3 起步時上緣 y 從 299 單調走到 411）。
      // 所以上緣的 +y 對應「遠離」，必須反號才符合既有約定。
      // （不用中心 y 的理由見檔頭：中心吃到被儀表板污染的高度。）
      dxRel: 0,
      dyRel: -kTop * dt,
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
