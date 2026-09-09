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

  // 車身參考區用「燈帶以下」的中央 —— 必須避開位於車後中線的第三剎車燈
  const by0 = Math.max(0, Math.round(h * cfg.bodyYTop));
  const by1 = Math.min(h, Math.max(by0 + 1, Math.round(h * cfg.bodyYBottom)));
  const hl = roiHist(data, w, 0, sw, y0, y1);
  const hr = roiHist(data, w, w - sw, w, y0, y1);
  const hb = roiHist(data, w, cx0, cx0 + cw, by0, by1);
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
    chGrid: chmslGrid(data, w, h, cfg),
    overexposed: tot > 0 ? over / tot : 0,
    luma: nAll > 0 ? luma / nAll : 0,
    n: nAll,
  };
}

/**
 * 第三剎車燈的搜尋網格：中央上部切成 cols×rows 格，每格取「最大紅度」。
 *
 * 取最大而不是平均：這顆燈很小（實測在 bbox 寬的 x 0.44~0.56、高的 y 0~0.10，
 * 佔整個 bbox 不到 2%），取平均會被周圍的暗車身稀釋掉。
 *
 * 搜尋範圍由法規決定而不是猜的：「應裝置於車後中線且其基準中心應高於
 * 煞車燈基準中心」→ 中央、上半部。
 *
 * 回傳的是原始網格，判定留給 BrakeLightDetector —— 因為那需要跨 tick 的
 * 位置追蹤與峰值/谷值，不屬於純函式。
 */
export function chmslGrid(data, w, h, cfg = CONFIG.brakeLight) {
  const c = cfg.chmsl;
  const x0 = Math.max(0, Math.round(w * (0.5 - c.searchXFrac / 2)));
  const x1 = Math.min(w, Math.round(w * (0.5 + c.searchXFrac / 2)));
  const y1 = Math.min(h, Math.max(2, Math.round(h * c.searchYFrac)));
  const vals = new Float32Array(c.cols * c.rows);
  for (let ry = 0; ry < c.rows; ry++) {
    const cy0 = Math.round(y1 * ry / c.rows), cy1 = Math.round(y1 * (ry + 1) / c.rows);
    for (let rx = 0; rx < c.cols; rx++) {
      const cx0 = x0 + Math.round((x1 - x0) * rx / c.cols);
      const cx1 = x0 + Math.round((x1 - x0) * (rx + 1) / c.cols);
      let mx = 0;
      for (let y = cy0; y < cy1; y++) {
        const row = y * w;
        for (let x = cx0; x < cx1; x++) {
          const i = (row + x) * 4;
          const v = redness(data[i], data[i + 1], data[i + 2]);
          if (v > mx) mx = v;
        }
      }
      vals[ry * c.cols + rx] = mx;
    }
  }
  // 每格中心的正規化座標（0~1，相對整個 bbox）——位置追蹤要用
  return {
    vals, cols: c.cols, rows: c.rows,
    cellW: (x1 - x0) / c.cols / w,
    cellH: y1 / c.rows / h,
    xOff: x0 / w,
  };
}

const fmt = (v) => (isFinite(v) ? v.toFixed(0) : '--');

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
    // 'on'      剎車中（動態範圍已解析，且位準在高檔）
    // 'lit'     有一對紅燈亮著，但還分不出是尾燈還是剎車燈
    // 'off'     位準明顯低於峰值 → 剎車已鬆開
    // 'unknown' 過曝，或看不到一對紅燈
    this.state = 'unknown';
    this.everOn = false;          // 是否曾確認過「有一對紅燈」
    this.coreL = 0; this.coreR = 0;       // 燈芯亮度峰值（決定面積門檻 θ）
    this.peakL = 0; this.peakR = 0;       // 「亮起來的面積」峰值 ← 判定 on/off 用這個
    this.areaL = 0; this.areaR = 0;       // 本 tick 的面積
    this.levelL = 0; this.levelR = 0;     // 本 tick 的位準（面積×亮度）
    // 位準的谷值 —— 與峰值一起決定動態範圍。Infinity = 還沒觀察到。
    // 用 Infinity 而不是 0 當初值：完全熄滅時位準就是 0，
    // 若用 0 當哨兵，最乾淨的那種落差反而會被當成「還沒觀察到」。
    this.floorL = Infinity; this.floorR = Infinity;
    this.offSince = 0;
    this.lastTs = 0;
    this.releaseTs = -Infinity;   // 最近一次確認「熄滅」的時刻
    this.offEdges = [];           // 近期 on→off 的時刻（判斷雙閃/方向燈用）
    this.blinking = false;
    this.resolved = false;   // 動態範圍是否已足以區分尾燈與剎車燈

    // ---- 第三剎車燈（車頂中央那一顆）----
    this.chPos = null;            // 追蹤到的位置（bbox 內的正規化座標）
    this.chPeak = 0;              // 這顆燈自己的亮度峰值
    this.chFloor = Infinity;      // 與谷值 —— 兩者的比值決定「能不能信」
    this.chVal = 0;
    this.chState = 'unknown';     // 'on' | 'off' | 'unknown'
    this.chPendingState = null;
    this.chPendingSince = 0;
    this.chPressTs = -Infinity;   // 最近一次「踩下」（起步前兆，領先約 9 秒）
    this.chUsable = false;        // 這台車看不看得到第三剎車燈
    this.chFar = false;           // 車太遠 → 暫時不用這條判據
    this.lastDetail = 'idle';
    this.lastStats = null;
  }

  /** 是否處於「已預備」狀態（剎車燈剛熄，起步很可能馬上發生） */
  primed(now) {
    return now - this.releaseTs <= this.cfg.brakeLight.priorValidMs;
  }

  /** 前車踩下剎車後尚在有效期內（自排打檔必須踩剎車 → 起步前兆，領先約 9 秒） */
  pressed(now) {
    return now - this.chPressTs <= this.cfg.brakeLight.chmsl.pressValidMs;
  }

  /**
   * 第三剎車燈判定。
   *
   * 兩個關鍵設計，都是被實測資料逼出來的：
   *
   * 1. **不跟空間鄰居比，跟這顆燈自己的過去比。**
   *    原本想用「中央格 / 左右鄰居」的空間對比做無狀態判定，看起來很乾淨。
   *    但 84 幀的實測顯示它直接重疊：燈的紅度在相鄰兩幀是 142 → 143（沒變），
   *    鄰居卻從 74 跳到 178，比值因此從 1.9 掉到 0.8 —— 判定翻面。
   *    原因是 YOLO 的框每幀都在抖（266×262 → 283×267），
   *    「bbox 上緣 12.5%」這條帶會蓋到車身的不同位置。
   *    **任何以 bbox 固定比例當基準的空間參考都會被框的抖動害死**
   *    （這是同一個坑的第二次：第一次是外側燈的 yTop=0.38 量到了保險桿反光片）。
   *
   * 2. **位置要追蹤。** 燈在車上是固定的，所以位置的穩定性本身就是一道驗證。
   *    只有在「明顯是一顆燈」（紅度遠高於網格中位數）時才更新追蹤位置，
   *    之後一律讀「追蹤位置上的值」而不是「網格最大值」——
   *    否則背景其他車的紅燈飄進搜尋區就會被誤認。
   */
  _updateChmsl(st, now, boxW) {
    const c = this.cfg.brakeLight.chmsl;
    const g = st.chGrid;
    const out = { usable: false, state: 'unknown', pressed: false, released: false, val: 0 };
    if (!g) return out;

    // 車太遠 → 這顆燈只剩幾個像素，搜尋區還會吃到背景。實測 84 幀裡
    // 唯一判錯的就是車駛遠之後的那一幀。
    if (boxW && boxW < c.minBoxW) {
      this.chFar = true;
      return out;
    }
    this.chFar = false;

    // ---- 找候選格 ----
    let mi = 0, mv = -1;
    for (let i = 0; i < g.vals.length; i++) if (g.vals[i] > mv) { mv = g.vals[i]; mi = i; }
    const sorted = Array.from(g.vals).sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)];
    const posOf = (idx) => ({
      x: g.xOff + (idx % g.cols + 0.5) * g.cellW,
      y: ((idx / g.cols) | 0) + 0.5,
    });
    const cand = posOf(mi);
    cand.y *= g.cellH;

    // 「這是一顆燈」而不是車身：紅度要遠高於網格中位數，且有絕對下限。
    // 這一關只管「要不要更新追蹤位置」，寧嚴勿寬 —— 鎖錯位置的代價很高。
    const looksLit = mv >= Math.max(c.litVsGrid * med, c.litMin);

    // 追蹤位置上的目前值
    const readAt = (pos) => {
      const rx = Math.min(g.cols - 1, Math.max(0,
        Math.round((pos.x - g.xOff) / g.cellW - 0.5)));
      const ry = Math.min(g.rows - 1, Math.max(0,
        Math.round(pos.y / g.cellH - 0.5)));
      return g.vals[ry * g.cols + rx];
    };

    // ---- 位置追蹤 ----
    if (looksLit) {
      if (!this.chPos) {
        this.chPos = { x: cand.x, y: cand.y };
      } else if (Math.abs(cand.x - this.chPos.x) <= c.posTol) {
        // 就在追蹤位置附近 → 微調（燈在車上不動，位置本身就是一道驗證）
        this.chPos.x = 0.8 * this.chPos.x + 0.2 * cand.x;
        this.chPos.y = 0.8 * this.chPos.y + 0.2 * cand.y;
      } else if (mv >= c.relockRatio * Math.max(readAt(this.chPos), 1)) {
        // 別處出現明顯更強的候選 → 重新鎖定。
        // 這是唯一的恢復路徑：實測曾在第三燈還沒亮時把位置鎖到尾燈上，
        // 之後真正的燈亮起也讀不到，因為它離追蹤位置太遠。
        this.chPos = { x: cand.x, y: cand.y };
        this.chPeak = 0;
        this.chFloor = Infinity;
        this.chState = 'unknown';
      }
    }

    // ---- 一律讀「追蹤位置上的值」，不是網格最大值 ----
    const val = this.chPos ? readAt(this.chPos) : mv;
    out.val = val;
    this.chVal = val;

    // ---- 這顆燈自己的峰值/谷值 ----
    if (looksLit && val > this.chPeak) this.chPeak = val;
    if (val < this.chFloor) this.chFloor = val;
    // 寫成不等式，這樣 floor = 0（完全熄滅）自然代表範圍無限大
    const resolved = this.chPeak > 0 && isFinite(this.chFloor)
      && this.chFloor <= this.chPeak / c.rangeMin;
    out.usable = resolved && !!this.chPos;
    if (!this.chPos) return out;

    // ---- 狀態機：一律跑，即使範圍還沒解析 ----
    // 理由：**轉換本身就是證據**。若鎖定時燈已經亮著（停在紅燈後方的常見情形），
    // 峰值 = 谷值、範圍未解析；但只要燈一暗下來，谷值就會掉到 0、
    // 範圍在同一個畫格立刻解析（0 ≤ peak/3）。
    // 若在這裡就 return，最重要的那個「鬆開」邊緣會被自己吞掉。
    // 邊緣事件仍然只在解析後回報（見下方），狀態則交由呼叫端依 usable 決定要不要採用。
    const want = val >= this.chPeak * c.onRatio ? 'on'
      : val <= this.chPeak * c.offRatio ? 'off' : this.chState;
    if (want !== this.chState) {
      if (this.chPendingState !== want) { this.chPendingState = want; this.chPendingSince = now; }
      if (now - this.chPendingSince >= c.confirmMs) {
        const prev = this.chState;
        this.chState = want;
        this.chPendingState = null;
        if (want === 'on' && prev !== 'on') { out.pressed = true; this.chPressTs = now; }
        if (want === 'off' && prev === 'on') { out.released = true; }
      }
    } else {
      this.chPendingState = null;
    }
    out.state = this.chState;
    // 範圍還沒解析 → 有可能只是中央上部有個一直紅著的東西（貼紙、反光），
    // 還不能當成剎車燈的邊緣事件回報
    if (!out.usable) { out.pressed = false; out.released = false; }
    return out;
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
    // boxW 用「影像座標」的框寬，不是裁切後的 —— 它是距離的代理，
    // 而裁切寬度被 maxSide 夾住之後就失去了距離資訊
    return this.updateFromStats(lampStats(img.data, w, h, b), now, box.w);
  }

  /**
   * 純判定：吃 lampStats 的輸出，輸出狀態與「剛剛熄滅」的邊緣事件。
   * 這裡沒有任何 DOM，也沒有任何絕對像素門檻 —— 全部是比值。
   */
  updateFromStats(st, now, boxW = Infinity) {
    const b = this.cfg.brakeLight;
    this.lastStats = st;

    // ---- 峰值參考值隨時間衰減（車換了、天色變了都該慢慢忘記）----
    if (this.lastTs) {
      const decay = Math.pow(0.5, (now - this.lastTs) / b.peakHalfLifeMs);
      this.peakL *= decay;
      this.peakR *= decay;
      this.coreL *= decay;
      this.coreR *= decay;
      // 谷值往上放（忘記舊的低點），與峰值往下衰減對稱
      if (isFinite(this.floorL)) this.floorL /= decay;
      if (isFinite(this.floorR)) this.floorR /= decay;
      this.chPeak *= decay;
      if (isFinite(this.chFloor)) this.chFloor /= decay;
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

    // ---- 第三剎車燈（優先）----
    // 放在過曝檢查之後：全白的畫面會讓網格紅度歸零，看起來像「熄滅」
    const ch = this._updateChmsl(st, now, boxW);
    this.chUsable = ch.usable;

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
    // 存起來給 UI —— 分析頻率低於畫面幀率，UI 不能只在有量測的那一幀才有值，
    // 否則面板會在「數值」與「--」之間跳（又是一種閃爍）
    this.levelL = lvlL; this.levelR = lvlR;

    // 第三剎車燈被找到，本身就證明「這是一個有剎車燈的車尾」——
    // 而且它比左右燈區的對比檢定可靠得多（實測那台車對比只有 0.72，
    // 因為中央參考區被第三剎車燈自己汙染了）。
    if (ch.usable || (this.chPos && this.chPeak > 0)) this.everOn = true;
    if (present) {
      this.everOn = true;
      // 峰值只在「確認有一對紅燈」時更新，免得雜訊或旁車的紅光拉高它
      if (lvlL > this.peakL) this.peakL = lvlL;
      if (lvlR > this.peakR) this.peakR = lvlR;
    }
    // 谷值不受 present 限制：它的語意是「這個區域曾經多暗」，與有沒有燈無關。
    // 燈全暗時 present 會（正確地）變成 false —— 若把谷值鎖在 present 裡面，
    // 最乾淨的那個低點永遠記錄不到，動態範圍就永遠解析不了。
    if (lvlL < this.floorL) this.floorL = lvlL;
    if (lvlR < this.floorR) this.floorR = lvlR;

    let released = false;
    const havePeak = this.peakL > 0 && this.peakR > 0;
    // 動態範圍是否足以區分「尾燈」與「剎車燈」
    // 寫成不等式而不是除法，這樣 floor = 0（完全熄滅）自然代表範圍無限大
    this.resolved = isFinite(this.floorL) && isFinite(this.floorR)
      && this.floorL <= this.peakL / b.onRange
      && this.floorR <= this.peakR / b.onRange;

    if (havePeak && present && lvlL >= this.peakL * b.onRatio
        && lvlR >= this.peakR * b.onRatio) {
      // 位準在自身高檔 —— 但只有在動態範圍已解析時才敢說是「剎車中」
      this.state = this.resolved ? 'on' : 'lit';
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

    // ---- 第三剎車燈可用時，它是權威 ----
    // 落差 70 倍 vs 外側燈的 1.5 倍；而且法規要求「續亮不得閃爍」，
    // 所以這條路徑連閃爍抑制都不需要（方向燈也不在中央上部）。
    if (ch.usable) {
      if (ch.state !== 'unknown') this.state = ch.state;
      released = ch.released;
    }

    if (released) this.releaseTs = now;

    this.lastDetail = `lvl=${lvlL.toFixed(0)}/${lvlR.toFixed(0)}`
      + ` peak=${this.peakL.toFixed(0)}/${this.peakR.toFixed(0)}`
      + ` floor=${fmt(this.floorL)}/${fmt(this.floorR)}`
      + ` area=${this.areaL.toFixed(2)}/${this.areaR.toFixed(2)}`
      + ` core=${this.coreL.toFixed(0)}/${this.coreR.toFixed(0)}`
      + ` c=${contrast.toFixed(2)} sym=${sym.toFixed(2)}`
      + ` ${this.state}${this.blinking ? ' BLINK' : ''}${present ? '' : ' no-pair'}`
      + `
      第三燈 ${ch.usable ? ch.state
          : this.chFar ? '車太遠'
          : !this.chPos ? '找不到'
          : '範圍未解析'}`
      + ` val=${this.chVal.toFixed(0)} peak=${this.chPeak.toFixed(0)}`
      + ` floor=${fmt(this.chFloor)}`
      + (this.chPos ? ` pos=${this.chPos.x.toFixed(2)},${this.chPos.y.toFixed(2)}` : '')
      + (this.pressed(now) ? ' 已踩下' : '');

    return {
      state: this.state,
      usable: true,
      released,
      blinking: this.blinking,
      primed: this.primed(now),
      everOn: this.everOn,
      present,
      resolved: this.resolved,
      contrast,
      sym,
      level: Math.min(lvlL, lvlR),
      // 第三剎車燈：pressed 是「暗了很久之後亮起」的邊緣 ——
      // 自排打檔必須踩剎車，所以它是起步的前兆（實測領先約 9 秒）
      chmsl: {
        usable: ch.usable, state: ch.state, val: ch.val,
        peak: this.chPeak, pos: this.chPos, far: this.chFar,
      },
      pressed: ch.pressed,
    };
  }

  debugLine() {
    return `brake ${this.lastDetail}`;
  }
}
