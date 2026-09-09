// =============================================
// 前車選取
// =============================================
// v6 用 MiDaS 深度挑「最前方」，有三個問題：
//   1. getDepthForBbox 取 bbox 內所有像素的平均 —— bbox 四角必然含背景，
//      遠車 bbox 小、背景佔比高，平均值被拉偏
//   2. MiDaS 輸出是 relative inverse depth，每幀 scale/shift 都不同
//   3. 每 tick 多花 80~250ms，把整個系統的節拍拖垮
//
// 改用透視幾何：「bbox 底邊越低 = 越近」。這在單目相機下是嚴格成立的
// （只要車輪貼地），免費、跨幀穩定，而且不需要任何模型。
//
// 走廊（corridor）取代 v6 的「中央 60% 硬閘」，且由地面平面幾何嚴格推導：
//   影像 y 處的地面距離   Z = f·h / (y − y_horizon)
//   橫向 X 公尺的成像寬   X_px = f·X / Z = X·(y − y_horizon) / h
//   → 焦距 f 消掉，走廊半寬只取決於「車道半寬 / 相機高度」這個比值。
// 於是走廊在地平線處自動收斂到 0、往畫面下方線性放寬，
// 而且參數是兩個真實可量的物理量，不是憑感覺的畫面百分比。
// 地平線位置由 IMU 重力向量推得（取代 v6 寫死的 0.45）。

import { clamp, iou } from '../util/math.js';

export class FrontCarSelector {
  constructor(cfg) {
    this.cfg = cfg;
    this.selectedId = null;
    // 車道中心的橫向位置（畫面比例）。手機常常沒有裝在正中央，
    // 所以用極慢的速率線上學習，並夾在 ±0.15 內防止跑掉。
    this.laneCenterX = 0.5;
    this.horizon = cfg.frontCar.horizonFallback;
    this.aspect = 9 / 16;       // vh/vw，每 tick 由實際影像尺寸更新

    // 自車結構黑名單。用「畫面上的區域」而不是 track id 記錄 ——
    // 引擎蓋在影像上的位置固定不變，而 track id 會隨偵測斷續不斷換新。
    this.egoRegions = [];       // [{ x, y, w, h }] 正規化 0~1
    this._frozenSince = new Map();   // trackId -> 開始「完全不動」的時刻

    // 族群的 w_px/Δy 樣本。地面幾何保證這個比值 = W_car/h_cam，
    // 與距離、焦距都無關，所以畫面上所有真實車輛會聚在同一個值上，
    // 而不站在地面上的自車結構是離群點。中位數線上學 → 不必寫死相機高度。
    this._ratioSamples = [];         // { v, id, ts }
    this._ratioLastTs = new Map();   // trackId -> 上次取樣時刻
  }

  reset() {
    this.selectedId = null;
    this._frozenSince.clear();
    this._ratioLastTs.clear();
    this._ratioSamples.length = 0;
  }

  /** 更新地平線估計（有 IMU 就用 IMU，沒有就用 config fallback） */
  setHorizon(ratio) {
    if (ratio !== null && isFinite(ratio)) {
      this.horizon = 0.9 * this.horizon + 0.1 * clamp(ratio, 0.1, 0.8);
    }
  }

  /**
   * 在畫面高度比例 yr 處，走廊的半寬（以畫面寬度為單位的比例）。
   *   半寬_px = (車道半寬 / 相機高度) · (y − y_horizon)
   * 焦距在推導中消掉了，所以只需要兩個物理量。
   * @param aspect vh / vw
   */
  halfWidthAt(yr, aspect = this.aspect) {
    const f = this.cfg.frontCar;
    const dy = Math.max(yr - this.horizon, 0);          // 以畫面高為單位
    const k = f.laneHalfWidthM / Math.max(f.cameraHeightM, 0.1);
    return k * dy * aspect;                             // 轉成以畫面寬為單位
  }

  /**
   * 走廊權重 0~1：在走廊內 = 1，走廊外以高斯衰減。
   * 用 bbox 底邊中心（車輪接地點）判斷，不是幾何中心 —— 這才是車在路面上的位置。
   */
  corridorWeight(box, vw, vh) {
    this.aspect = vh / vw;
    const bx = (box.x + box.w / 2) / vw;
    const byr = (box.y + box.h) / vh;
    const hw = this.halfWidthAt(byr, this.aspect);
    const d = Math.abs(bx - this.laneCenterX) / Math.max(hw, 1e-3);
    if (d <= 1) return 1;
    const t = (d - 1) / this.cfg.frontCar.corridorSoftness;
    return Math.exp(-0.5 * t * t);
  }

  /** 硬性資格：只保留幾何上真的不可能是前車的排除條件 */
  isCandidate(box, vw, vh) {
    const f = this.cfg.frontCar;
    // (1) 底邊必須在地平線下方（車輪不可能在地平線之上）
    if ((box.y + box.h) / vh <= this.horizon) return false;
    // (2) 面積太小 → 太遠，量測不可靠
    if ((box.w * box.h) / (vw * vh) < f.minAreaRatio) return false;
    // (3) 命中已學到的自車結構區域（由「行駛中卻完全不動」這個物理性質學來）
    if (this.isEgoStructure(box, vw, vh)) return false;
    // (4) 走廊權重過低（注意：車道線模型只做軟性加權，不參與這個硬閘）
    if (this.corridorWeight(box, vw, vh) < f.minCorridorWeight) return false;
    return true;
  }

  /**
   * 幾何可信度 0~1：「這個框像不像一台站在地面上的車」。
   * 軟性加權，不做硬性排除 —— 因為斜看的大車、被裁切的近車都會落在
   * 區間邊緣，硬排除的代價（漏掉真前車）比誤選的代價高。
   *
   * 兩項都由實體尺寸推導，沒有任何畫面比例常數：
   *   (a) 車尾長寬比：車尾寬 1.4~2.6m、高 1.2~3.2m
   *   (b) w_px/Δy = W_car/h_cam（焦距與距離都消掉了）—— 只查高側離群，
   *       因為自車結構偏高、機車偏低，單側檢定才不會誤殺機車
   */
  plausibility(box, vw, vh) {
    const p = this.cfg.frontCar.plausibility;
    let wgt = 1;

    // (a) 長寬比
    const ar = box.w / Math.max(box.h, 1e-3);
    if (ar < p.aspectMin || ar > p.aspectMax) {
      const d = ar < p.aspectMin ? p.aspectMin - ar : ar - p.aspectMax;
      wgt *= Math.exp(-0.5 * Math.pow(d / p.aspectSoftness, 2));
    }

    // (b) w/Δy 的高側離群。參考值 = 車寬 / 相機高度（兩個公尺數，焦距已消掉）
    const r = this.groundRatio(box, vw, vh);
    if (r !== null) {
      const f = this.cfg.frontCar;
      const expected = f.vehicleWidthM / Math.max(f.cameraHeightM, 0.1);
      const over = r / (expected * p.ratioOutlierFactor);
      if (over > 1) {
        wgt *= Math.exp(-0.5 * Math.pow((over - 1) / p.ratioSoftness, 2));
      }
    }
    return wgt;
  }

  /** w_px / (y_bottom − y_horizon)。地面上的車輛此值恆為 W_car/h_cam。 */
  groundRatio(box, vw, vh) {
    const dy = (box.y + box.h) - this.horizon * vh;
    if (!(dy > 4)) return null;             // 太靠近地平線 → 分母不可靠
    return box.w / dy;
  }

  /** 族群比值的中位數 —— 目前只用於觀察（除錯面板），不參與判定 */
  ratioMedian() {
    const p = this.cfg.frontCar.plausibility;
    const n = this._ratioSamples.length;
    if (n < p.ratioMinSamples) return null;
    const v = this._ratioSamples.map((s) => s.v).sort((a, b) => a - b);
    return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
  }

  /**
   * 收集族群比值樣本。
   * 同一個 track 最快每 ratioSampleGapMs 貢獻一次 —— 否則畫面上持續存在的
   * 單一目標（正好就是引擎蓋假框）會用樣本數主導中位數，把離群值變成中心。
   * 已被列入自車結構黑名單的框不取樣。
   */
  noteRatioSamples(tracks, vw, vh, now) {
    const p = this.cfg.frontCar.plausibility;
    for (const tr of tracks) {
      const last = this._ratioLastTs.get(tr.id) || 0;
      if (now - last < p.ratioSampleGapMs) continue;
      const box = tr.boxAt(now);
      if (this.isEgoStructure(box, vw, vh)) continue;
      const r = this.groundRatio(box, vw, vh);
      if (r === null || !isFinite(r)) continue;
      this._ratioLastTs.set(tr.id, now);
      this._ratioSamples.push({ v: r, id: tr.id, ts: now });
      if (this._ratioSamples.length > p.ratioMaxSamples) this._ratioSamples.shift();
    }
  }

  /** 是否落在已學到的自車結構區域上 */
  isEgoStructure(box, vw, vh) {
    if (!this.egoRegions.length) return false;
    const n = { x: box.x / vw, y: box.y / vh, w: box.w / vw, h: box.h / vh };
    const th = this.cfg.frontCar.egoStructure.matchIou;
    for (const r of this.egoRegions) if (iou(n, r) > th) return true;
    return false;
  }

  /**
   * 學習自車結構。
   * 只在「自車確定在行駛」時呼叫才有意義：此時畫面上任何完全靜止的框，
   * 都只能是自車的一部分（路邊停的車在自車前進時 bbox 會持續放大）。
   * 紅燈停車時前車本來就不動，若在那時學習會把真前車列入黑名單。
   */
  learnEgoStructure(tracks, vw, vh, now, egoMoving) {
    const es = this.cfg.frontCar.egoStructure;
    if (!egoMoving) { this._frozenSince.clear(); return; }

    const alive = new Set();
    for (const tr of tracks) {
      alive.add(tr.id);
      const frozen = Math.abs(tr.kf.logScaleRate) < es.frozenLogScaleRate &&
                     Math.abs(tr.kf.vy) < es.frozenVyPxPerS;
      if (!frozen) { this._frozenSince.delete(tr.id); continue; }
      const since = this._frozenSince.get(tr.id);
      if (since === undefined) { this._frozenSince.set(tr.id, now); continue; }
      if (now - since < es.learnWhileMovingMs) continue;

      const box = tr.boxAt(now);
      const n = { x: box.x / vw, y: box.y / vh, w: box.w / vw, h: box.h / vh };
      if (this.egoRegions.some((r) => iou(n, r) > es.matchIou)) continue;
      this.egoRegions.push(n);
      if (this.egoRegions.length > es.maxRegions) this.egoRegions.shift();
      if (tr.id === this.selectedId) this.selectedId = null;   // 選錯了，立刻放手
    }
    for (const id of this._frozenSince.keys()) if (!alive.has(id)) this._frozenSince.delete(id);
  }

  /**
   * 從 track 清單中選出前車。
   * @param tracks  已確認的車輛 track
   * @param now     現在時刻（用於 KF 外推）
   * @param opts.laneWeightFn (box) => 0~1 或 null，車道線的軟性加權
   * @param opts.lampWeightFn (trackId) => 0~1，剎車燈的正向證據加權
   * @returns 選中的 track 或 null
   */
  select(tracks, vw, vh, now, opts = {}) {
    const { laneWeightFn = null, lampWeightFn = null } = opts;
    let best = null, bestScore = -1;
    let current = null, currentScore = -1;

    this.noteRatioSamples(tracks, vw, vh, now);

    for (const tr of tracks) {
      const box = tr.boxAt(now);
      if (!this.isCandidate(box, vw, vh)) continue;

      const cw = this.corridorWeight(box, vw, vh);
      const lw = laneWeightFn ? laneWeightFn(box) : null;
      const laneW = (lw === null || lw === undefined) ? 1 : (0.5 + 0.5 * lw);
      const plaus = this.plausibility(box, vw, vh);
      // 紅燈停等時前車的剎車燈幾乎必然亮著 —— 所以「看得到剎車燈」是比任何
      // 幾何規則都直接的正向證據，而且與手機安裝方式完全無關。
      // 預設中性（1），只有在「觀察夠久卻始終沒看到燈」時才降權。
      const lampW = lampWeightFn ? lampWeightFn(tr.id) : 1;
      // 新鮮度：coasting 的框是預測而不是量測，兩者同時存在時量測該贏。
      const age = Math.max(0, now - tr.lastSeenTs);
      const freshW = Math.pow(0.5, age / this.cfg.frontCar.plausibility.staleHalfLifeMs);

      // 底邊越低 = 越近（透視幾何）。正規化到 0~1。
      const proximity = clamp(
        ((box.y + box.h) / vh - this.horizon) / Math.max(1 - this.horizon, 1e-3), 0, 1
      );
      const score = proximity * cw * laneW * plaus * lampW * freshW;

      if (tr.id === this.selectedId) { current = tr; currentScore = score; }
      if (score > bestScore) { bestScore = score; best = tr; }
    }

    // 遲滯：已選中的目標要明顯輸給對手才換人，避免在兩台車之間反覆跳動
    // （v6 的 ID switch 會導致鎖到旁車道車起步 → 誤判）
    if (current && bestScore < currentScore * 1.25) {
      best = current;
      bestScore = currentScore;
    }

    if (!best) { this.selectedId = null; return null; }
    this.selectedId = best.id;

    // 用選中的前車極慢地校正車道中心（修正手機沒裝在正中央）
    const box = best.boxAt(now);
    const bx = (box.x + box.w / 2) / vw;
    this.laneCenterX = clamp(0.999 * this.laneCenterX + 0.001 * bx, 0.35, 0.65);

    return best;
  }

  debugLine() {
    const med = this.ratioMedian();
    return `horizon=${this.horizon.toFixed(2)} laneCx=${this.laneCenterX.toFixed(3)}`
      + ` egoRegions=${this.egoRegions.length}`
      + ` w/dy=${med === null ? '--' : med.toFixed(2)}(n=${this._ratioSamples.length})`;
  }
}
