// =============================================
// 剎車燈狀態（起步警示的「快路徑」）
// =============================================
// 為什麼要有這條路徑：
//   所有基於運動的量測（尺度變化率、光流、深度）都必須「等車真的動了」
//   才會有訊號。但駕駛鬆開剎車踏板到車輛實際移動之間有 0.3~1 秒，
//   而剎車燈在鬆踏板的那一瞬間就熄了 —— 這是唯一能讓警示
//   「本質上變快」而不是「量得更準」的訊號源。
//
//   訊噪比也完全不同量級：5~10m 的剎車燈在影像上是數十像素的飽和紅光，
//   而同一情境下 240ms 基線的尺度變化只有 2.4%（要靠 80 個點做最小平方
//   擬合才勉強看得到）。
//
// 為什麼不用絕對色彩門檻：
//   紅綠燈那條路（trafficLight.js）用的是絕對門檻 `r > 150 && r > g*1.45`
//   加上亮度/飽和度硬閘 `v < 0.45 || sat < 0.25`。那是「絕對分類」問題
//   （這團紅色是紅燈？落日？紅色招牌？），所以只能用絕對門檻，
//   而它換個曝光、換個燈型就失效。
//
//   剎車燈不是分類問題，是**同一台被追蹤車輛、幾秒內的相對變化**：
//   我們不需要知道「多紅才算剎車燈」，只需要知道「這個區域的紅度
//   比一秒前掉了多少」。所以這裡的判定量全部是比值：
//     * 燈區紅度 / 車身紅度        → 對「紅色車身」免疫
//     * 目前紅度 / 自己的歷史峰值  → 對曝光、增益、日夜差異免疫
//
// 失效時必須回報 'unknown' 而不是 'off'：
//   夕陽直射燈罩會讓 R、G、B 全部飽和成白塊，紅度反而歸零 ——
//   若把它當成「熄燈」就會在逆光時誤報。這與 egoMotion 的三態設計同源：
//   「不知道」是一個必須明確存在的狀態。

import { CONFIG } from '../config.js';

/** 紅度：紅通道超出另兩個通道的量。車尾燈是自發紅光，這個量最直接。 */
function redness(r, g, b) {
  const m = g > b ? g : b;
  const d = r - m;
  return d > 0 ? d : 0;
}

const BINS = 32, BIN_W = 256 / BINS;

/**
 * 掃一個矩形區域，把紅度做成 32 格直方圖。
 * 用直方圖而不是「收集所有值再排序」有兩個好處：
 *   1. 不配置陣列、不排序 —— 每 tick 省下數千次比較
 *   2. 同一次掃描就能導出兩個判定量：燈芯亮度（前 k% 平均）與
 *      「紅度超過 θ 的面積比」。後者是夜間唯一有效的量，見 config 的量測數據。
 */
function roiHist(data, w, x0, x1, y0, y1) {
  const hist = new Int32Array(BINS);
  let n = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) {
      const i = (row + x) * 4;
      hist[(redness(data[i], data[i + 1], data[i + 2]) / BIN_W) | 0]++;
      n++;
    }
  }
  return { hist, n };
}

/** 前 kFrac 比例最紅像素的平均紅度（「燈芯亮度」） */
export function topKFromHist(hist, n, kFrac) {
  const k = Math.max(4, Math.round(n * kFrac));
  let cnt = 0, sum = 0;
  for (let b = BINS - 1; b >= 0 && cnt < k; b--) {
    const take = Math.min(hist[b], k - cnt);
    sum += take * (b * BIN_W + BIN_W / 2);
    cnt += take;
  }
  return cnt ? sum / cnt : 0;
}

/** 紅度 ≥ θ 的像素佔比（「亮起來的面積」），邊界格做線性內插 */
export function areaAboveFromHist(hist, n, theta) {
  if (!n) return 0;
  const b0 = Math.max(0, Math.min(BINS - 1, Math.floor(theta / BIN_W)));
  let c = 0;
  for (let b = BINS - 1; b > b0; b--) c += hist[b];
  const frac = 1 - (theta - b0 * BIN_W) / BIN_W;
  c += hist[b0] * Math.max(0, Math.min(1, frac));
  return c / n;
}

/**
 * 純函式：從 RGBA 像素算出三個區域的紅度統計。
 * 不接觸任何 DOM，所以可以在 node 裡用合成影像做單元測試
 * （方向燈、雙閃、紅色車身、逆光四種干擾都能離線驗證）。
 *
 * @param data RGBA Uint8ClampedArray（長度 = w·h·4）
 * @returns { left, right, body, overexposed, luma, n }
 */
export function lampStats(data, w, h, cfg = CONFIG.brakeLight) {
  const y0 = Math.max(0, Math.round(h * cfg.yTop));
  const y1 = Math.min(h, Math.max(y0 + 1, Math.round(h * cfg.yBottom)));
  const sw = Math.max(1, Math.round(w * cfg.sideFrac));
  const cw = Math.max(1, Math.round(w * cfg.centerFrac));
  const cx0 = Math.max(0, Math.round((w - cw) / 2));

  const hl = roiHist(data, w, 0, sw, y0, y1);
  const hr = roiHist(data, w, w - sw, w, y0, y1);
  const hb = roiHist(data, w, cx0, cx0 + cw, y0, y1);
  const left = topKFromHist(hl.hist, hl.n, cfg.topKFrac);
  const right = topKFromHist(hr.hist, hr.n, cfg.topKFrac);
  const body = topKFromHist(hb.hist, hb.n, cfg.topKFrac);

  // 過曝：三通道都貼頂 → 資訊已經被裁掉，紅度會假性歸零
  let over = 0, tot = 0, luma = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = (row + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const mn = Math.min(r, Math.min(g, b));
      if (x < sw || x >= w - sw) { if (mn >= 240) over++; tot++; }
      luma += (r + g + b) / 3;
    }
  }
  const nAll = (y1 - y0) * w;

  return {
    left, right, body,
    // 直方圖交給 BrakeLightDetector 算「面積」—— 門檻 θ 取決於它保存的
    // 燈芯亮度峰值，那是跨 tick 的狀態，不屬於這個純函式
    histL: hl.hist, histR: hr.hist, nL: hl.n, nR: hr.n,
    overexposed: tot > 0 ? over / tot : 0,
    luma: nAll > 0 ? luma / nAll : 0,
    n: nAll,
  };
}

export class BrakeLightDetector {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.canvas = typeof document !== 'undefined'
      ? document.createElement('canvas')
      : (typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(32, 32) : null);
    this.ctx = this.canvas ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
    this.reset();
  }

  reset() {
    this.state = 'unknown';       // 'on' | 'off' | 'unknown'
    this.everOn = false;          // 是否曾確認過「有一對紅燈」
    this.coreL = 0; this.coreR = 0;       // 燈芯亮度峰值（決定面積門檻 θ）
    this.peakL = 0; this.peakR = 0;       // 「亮起來的面積」峰值 ← 判定 on/off 用這個
    this.areaL = 0; this.areaR = 0;       // 本 tick 的面積
    this.offSince = 0;
    this.lastTs = 0;
    this.releaseTs = -Infinity;   // 最近一次確認「熄滅」的時刻
    this.offEdges = [];           // 近期 on→off 的時刻（判斷雙閃/方向燈用）
    this.blinking = false;
    this.lastDetail = 'idle';
    this.lastStats = null;
  }

  /** 是否處於「已預備」狀態（剎車燈剛熄，起步很可能馬上發生） */
  primed(now) {
    return now - this.releaseTs <= this.cfg.brakeLight.priorValidMs;
  }

  /** 先驗對數勝算比：ln( P(熄燈│即將起步) / P(熄燈│不起步) ) */
  priorLlr() {
    const b = this.cfg.brakeLight;
    return Math.log(Math.max(b.pOffGivenDepart, 1e-6) / Math.max(b.pOffGivenStay, 1e-6));
  }

  /**
   * 影像端封裝：裁切目標框 → RGBA → updateFromStats。
   * @returns updateFromStats 的回傳
   */
  update(source, box, now) {
    const b = this.cfg.brakeLight;
    if (!this.ctx || !box || box.w < 24 || box.h < 16) {
      this.lastDetail = 'too-small';
      return { state: 'unknown', released: false, usable: false };
    }
    const scale = Math.min(1, b.maxSide / Math.max(box.w, box.h));
    const w = Math.max(8, Math.round(box.w * scale));
    const h = Math.max(8, Math.round(box.h * scale));
    this.canvas.width = w;
    this.canvas.height = h;
    let img;
    try {
      this.ctx.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, w, h);
      img = this.ctx.getImageData(0, 0, w, h);
    } catch (e) {
      this.lastDetail = 'capture-fail';
      return { state: 'unknown', released: false, usable: false };
    }
    return this.updateFromStats(lampStats(img.data, w, h, b), now);
  }

  /**
   * 純判定：吃 lampStats 的輸出，輸出狀態與「剛剛熄滅」的邊緣事件。
   * 這裡沒有任何 DOM，也沒有任何絕對像素門檻 —— 全部是比值。
   */
  updateFromStats(st, now) {
    const b = this.cfg.brakeLight;
    this.lastStats = st;

    // ---- 峰值參考值隨時間衰減（車換了、天色變了都該慢慢忘記）----
    if (this.lastTs) {
      const decay = Math.pow(0.5, (now - this.lastTs) / b.peakHalfLifeMs);
      this.peakL *= decay;
      this.peakR *= decay;
      this.coreL *= decay;
      this.coreR *= decay;
    }
    this.lastTs = now;

    // ---- 過曝 → 明確回報「不知道」，絕不回報「熄滅」----
    if (st.overexposed > b.overexposedFrac) {
      this.state = 'unknown';
      this.offSince = 0;
      this.lastDetail = `overexposed ${(st.overexposed * 100) | 0}%`;
      // usable:false → 呼叫端不該把這一段時間算成「沒看到剎車燈」的證據
      return { state: 'unknown', released: false, usable: false, primed: this.primed(now) };
    }

    // ---- 三個量，各有明確的分工 ----
    // (1) 對比 = 燈區紅度 / 車身紅度 → 紅色車身的漆面紅度會被除掉。
    //     +6 是 8-bit 量化與感測器雜訊的底線，避免暗處除以 ~0 得到假高對比。
    const weak = Math.min(st.left, st.right);
    const contrast = weak / (st.body + 6);
    // (2) 對稱性：兩顆尾燈同亮同熄。方向燈是單側 → 被這一項否決。
    const sym = Math.abs(Math.log((st.left + 1) / (st.right + 1)));
    // 「有一對對稱紅燈」≠「剎車燈亮」：
    //   夜間尾燈本來就亮著，剎車燈是同一燈室變更亮。所以 present 只用來
    //   (a) 建立峰值參考 (b) 當作前車選取的正向證據（自車結構永遠不會有）。
    const present = contrast >= b.minContrast && sym <= b.symmetryTol;

    // (3) 位準 = 面積 × 燈芯亮度，物理意義是「這顆燈打進影像的紅光總通量」。
    //     為什麼不能只用亮度：夜間相機的紅通道在剎車燈與尾燈下都飽和（貼到 255），
    //     峰值亮度在數學上被裁掉了 —— 實車量測熄/亮只掉到 0.64~0.76，
    //     永遠達不到 offRatio。
    //     為什麼不能只用面積：面積會變是因為「更亮 → 光暈更大 → 燈殼被照亮的
    //     範圍更大」，這對共用同一燈室的車款成立（實車量測 0.22~0.30），
    //     但若某車款的剎車燈是獨立燈泡、亮度變而面積幾乎不變，面積就失效。
    //     取乘積 = 對通量積分，兩種情形都涵蓋，而且飽和不會把積分裁掉。
    //     面積門檻 θ 取「燈芯亮度峰值」的一個比例 → 仍是相對量，非絕對像素值。
    if (present) {
      // 燈比史上最亮 → 重新建立基準（θ 變了，舊的位準峰值就沒有可比性）
      if (st.left > this.coreL * 1.05) { this.coreL = st.left; this.peakL = 0; }
      if (st.right > this.coreR * 1.05) { this.coreR = st.right; this.peakR = 0; }
    }
    const thL = b.areaThetaFrac * Math.max(this.coreL, st.left);
    const thR = b.areaThetaFrac * Math.max(this.coreR, st.right);
    this.areaL = areaAboveFromHist(st.histL, st.nL, thL);
    this.areaR = areaAboveFromHist(st.histR, st.nR, thR);
    const lvlL = this.areaL * st.left;
    const lvlR = this.areaR * st.right;

    if (present) {
      this.everOn = true;
      if (lvlL > this.peakL) this.peakL = lvlL;
      if (lvlR > this.peakR) this.peakR = lvlR;
    }

    let released = false;
    const havePeak = this.peakL > 0 && this.peakR > 0;

    if (havePeak && present && lvlL >= this.peakL * b.onRatio
        && lvlR >= this.peakR * b.onRatio) {
      this.state = 'on';
      this.offSince = 0;
    } else if (havePeak) {
      const downL = lvlL < this.peakL * b.offRatio;
      const downR = lvlR < this.peakR * b.offRatio;
      if (downL && downR) {
        // 兩側同時變暗才算熄滅。只有一側掉 → 機車鑽車縫遮住一顆燈，
        // 或是方向燈造成的單側變化 → 不是鬆剎車。
        if (!this.offSince) this.offSince = now;
        if (now - this.offSince >= b.offConfirmMs && this.state !== 'off') {
          this.state = 'off';
          this.offEdges.push(now);
          released = true;
        }
      } else {
        this.offSince = 0;
      }
    }

    // ---- 閃爍抑制：雙閃/方向燈會產生週期性的 on→off ----
    const cut = now - b.blinkWindowMs;
    this.offEdges = this.offEdges.filter((t) => Math.abs(t) >= cut);
    const cycles = this.offEdges.filter((t) => t > 0).length;
    this.blinking = cycles >= b.blinkMinCycles;
    if (this.blinking) released = false;

    if (released) this.releaseTs = now;

    this.lastDetail = `lvl=${lvlL.toFixed(0)}/${lvlR.toFixed(0)}`
      + ` peak=${this.peakL.toFixed(0)}/${this.peakR.toFixed(0)}`
      + ` area=${this.areaL.toFixed(2)}/${this.areaR.toFixed(2)}`
      + ` core=${this.coreL.toFixed(0)}/${this.coreR.toFixed(0)}`
      + ` c=${contrast.toFixed(2)} sym=${sym.toFixed(2)}`
      + ` ${this.state}${this.blinking ? ' BLINK' : ''}${present ? '' : ' no-pair'}`;

    return {
      state: this.state,
      usable: true,
      released,
      blinking: this.blinking,
      primed: this.primed(now),
      everOn: this.everOn,
      present,
      contrast,
      sym,
      level: Math.min(lvlL, lvlR),
    };
  }

  debugLine() {
    return `brake ${this.lastDetail}`;
  }
}
