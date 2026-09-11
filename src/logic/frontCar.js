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
    // 選中目標最後的框。track id 斷掉重建時（實測 37 秒內新建 119 個 track、
    // 換手 35 次），遲滯若只認 id 就會歸零 —— 任何候選都能立刻搶走目標，
    // 畫面上的紅框跟著跳一次大小。幾何上蓋在原框上的新 track 應該
    // **繼承遲滯**：id 是實作細節，同一台車才是判準（與 sameTargetIou 同理）。
    this.selectedBox = null;
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
    this.selectedBox = null;
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
    // (0) 左右兩緣同時被畫面裁掉的框，不能當目標。
    //
    // 這不是幾何猜測，是可量測性論證（與 bboxScale 的 width-censored 同一條
    // 規則、同一個容差）：這種框的寬度被畫面卡死，尺度變化率量不到 ——
    // 一個**結構上不可能產出起步量測**的目標，選了只有壞處：它會佔住
    // 光流與剎車燈這兩條昂貴路徑，讓真前車永遠輪不到。
    //
    // 2026-09-11 手持實測：自車儀表板被 DETR 認成 car 的框 5/5 幀都是
    // 左右貼邊（x=0 且 x+w=W），而真前車 21/21 幀都不是。
    // 真前車近到佔滿整個畫面寬時也會被此規則排除 —— 那個距離下本來就
    // 什麼都量不了，排除是正確行為（tracker 會 coast，離開後自動恢復）。
    {
      const clip = this.cfg.bboxScale.edgeClipPx;
      if (box.x <= clip && box.x + box.w >= vw - clip) return false;
    }
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

    // (c) 底邊被畫面裁掉的框：低側檢定（只有這種框才做）
    //
    // 為什麼一般情況不查低側：機車比汽車窄，w/Δy 天生偏低，查低側會誤殺它。
    // 但底邊被畫面下緣裁掉是一個完全不同的狀況 —— 接地點在畫面外，
    // 代表這個物體「至少有那麼近」，於是它的寬度有一個**下限**：
    //
    //   接地點恰在畫面最下緣時   Δy_max = vh − y_horizon
    //   該距離下的車寬          w_min  = (W_car / h_cam) · Δy_max
    //
    // 焦距一樣在推導中消掉。實測這支影片（588x980、地平線 0.45）：
    //   汽車 w_min = 809 px，機車（0.8m）w_min = 359 px —— 都比畫面 588 寬還大。
    // 也就是說：**任何底邊被裁掉的真實車輛，都會寬到接近或超過整個畫面。**
    // 一個貼著畫面下緣、卻只有兩三百 px 寬的框，在幾何上不可能站在地面上
    // —— 那是方向盤、儀表板、A 柱反光這類黏在相機上的東西。
    //
    // 仍然做成軟性加權而不是硬性排除：地平線在沒有 IMU 時是 fallback 值，
    // 估錯了會讓 w_min 整個偏掉，不該由它一票否決。
    const clipped = (box.y + box.h) >= vh - p.bottomClipPx;
    if (clipped) {
      const f = this.cfg.frontCar;
      const dyMax = vh - this.horizon * vh;
      const wMin = (f.vehicleWidthM / Math.max(f.cameraHeightM, 0.1)) * dyMax;
      // 容許實際寬度只有理論下限的 clippedWidthFactor 倍（地平線可能估偏）
      const under = (wMin * p.clippedWidthFactor) / Math.max(box.w, 1);
      if (under > 1) {
        wgt *= Math.exp(-0.5 * Math.pow((under - 1) / p.clippedSoftness, 2));
      }

      // (d) 同樣只對底邊被裁掉的框：查 h/Δy 的**低側**。
      //
      //   由底邊：Z = f·h_cam / Δy      由高度：Z = f·H_obj / h_px
      //   相除 → h_px/Δy = H_obj/h_cam   ← 焦距又消掉了
      //
      // 對底邊被裁的框，這個檢定的方向是**安全**的：遮擋/裁切只會讓 Δy
      // 偏小、比值**偏高** —— 低側異常不可能是遮擋造成的。唯一會把比值
      // 壓低的是上緣也被畫面裁掉，所以那種框跳過不查。
      //
      // 2026-09-11 實測的分布（也是選這個界線的依據）：
      //   假框：儀表板 0.86~0.93、引擎蓋 0.69、儀表板上的車圖示 0.14
      //   真車：≥ 1.17（車高至少 1.4m ÷ 相機高 1.2m），底邊被裁只會更高
      //        （3m 近車實測 1.28、手持前車 1.53~2.47）
      // clippedHeightFactor=0.85 把界線放在 1.0 —— 假框簇上限 0.93 與
      // 真車下限 1.17 的正中間，兩側各留約 10% 餘裕吸收相機高度的個體差。
      const topClipped = box.y <= p.bottomClipPx;
      const dy = (box.y + box.h) - this.horizon * vh;
      if (!topClipped && dy > 4) {
        const hMin = (f.vehicleHeightMinM / Math.max(f.cameraHeightM, 0.1)) * dy;
        const underH = (hMin * p.clippedHeightFactor) / Math.max(box.h, 1);
        if (underH > 1) {
          wgt *= Math.exp(-0.5 * Math.pow((underH - 1) / p.clippedSoftness, 2));
        }
      }
    }
    // 總懲罰有下限 —— 與 SPRT 的 zClamp 同一個哲學、同一個理由：
    // 這些檢定的輸入（bbox 底邊、地平線估計）帶著非高斯的模型誤差，
    // 高斯尾巴給出的 e^-26 那種數字不是「26σ 的證據」，是垃圾輸入被
    // 當成真值。2026-09-11 實測：真前車的接地點被儀表板遮住 → w/Δy
    // 暴衝到 5.5 → 這裡回傳 0.000 → **真前車在排序裡被無限否決**，
    // 反而輸給儀表板上 65px 的車圖示。軟性證據可以降權，不可以有
    // 無限否決權 —— 一票否決的權力只留給 isCandidate 的可量測性論證。
    return Math.max(wgt, p.vetoFloor);
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
    const { laneWeightFn = null, lampWeightFn = null, detIntervalMs = 0 } = opts;
    let best = null, bestScore = -1;
    let current = null, currentScore = -1;
    // 遲滯的幾何繼承人：selectedId 已死，但框蓋在原目標框上的候選。
    // 沒有這一層，track id 一斷（實測 37 秒內新建 119 個）遲滯就歸零，
    // 任何候選都能立刻搶走目標 —— 紅框跟著跳一次大小。
    let heir = null, heirScore = -1, heirIou = 0;

    this.noteRatioSamples(tracks, vw, vh, now);

    // 新鮮度的半衰期以**實測偵測週期**為下限：寫死 400ms 在偵測 4Hz 的
    // 實機上，目標只要漏掉一次偵測就被打折到 0.65，任何新鮮的鄰居只要
    // 1.25 倍遲滯就搶走目標 —— 這是換手震盪的引擎之一。
    // 「stale」的意義本來就是「比一個偵測週期舊」，不是「比 400ms 舊」。
    const staleHalfLife = Math.max(this.cfg.frontCar.plausibility.staleHalfLifeMs, detIntervalMs);

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
      const freshW = Math.pow(0.5, age / staleHalfLife);

      // 「最近」的判據改用**影像寬度**，不再用底邊高度。
      //
      // 底邊那條路（「底邊越低 = 越近」）在支架工況下是嚴格成立的透視幾何，
      // 但 2026-09-11 手持實測證明它的前提「bbox 底邊 = 輪胎接地點」會被
      // 自車儀表板遮擋打破：真前車的接地點藏在儀表板後面（底邊 y≈620、
      // 儀表板上緣 y≈624），而內裝假框的底邊恰好就在畫面最下緣 ——
      // 於是被污染的量測給了假目標理論最大分（1.0）、給真前車 0.53。
      //
      // 寬度沒有這個弱點：
      //   * 同車道中，越近的車影像越寬（w_px = f·W_car/Z）—— 對「排序」而言
      //     焦距與實際車寬只是共同倍率，同類物體比大小時自然消掉
      //   * 遮擋只裁掉框的**下緣**，左右緣不受影響（實測真前車靜止時
      //     寬度變異 0.78%、高度 22.6%）
      //   * 不含 cameraHeightM、不含地平線 —— 手持晃動下沒有可以猜錯的參數
      // 代價：跨「車 vs 機車」比寬度會偏袒車。可接受 —— 同距離下擋在
      // 正前方的若是機車，車道權重與長寬比先驗仍在，且錯選成更寬的車
      // 只是保守（量一台更遠的車），不是危險方向。
      const proximity = clamp(box.w / vw, 0, 1);
      const score = proximity * cw * laneW * plaus * lampW * freshW;

      if (tr.id === this.selectedId) { current = tr; currentScore = score; }
      else if (this.selectedBox) {
        // 幾何繼承人：id 不同但框蓋在原目標框上 → 同一台車換了 id
        const ov = iou(box, this.selectedBox);
        if (ov >= this.cfg.tracker.sameTargetIou && ov > heirIou) {
          heirIou = ov; heir = tr; heirScore = score;
        }
      }
      if (score > bestScore) { bestScore = score; best = tr; }
    }

    // id 死了但幾何繼承人在 → 繼承目標身分（含遲滯保護）。
    // id 是實作細節，同一台車才是判準 —— 與 pipeline 的 sameTargetIou
    // 換目標判定同一個邏輯，差別是這裡連**遲滯**都一起繼承，
    // 否則每次 id 斷掉，任何候選都能免遲滯搶走目標。
    if (!current && heir) { current = heir; currentScore = heirScore; }

    // 遲滯：已選中的目標要明顯輸給對手才換人，避免在兩台車之間反覆跳動
    // （v6 的 ID switch 會導致鎖到旁車道車起步 → 誤判）
    if (current && bestScore < currentScore * 1.25) {
      best = current;
      bestScore = currentScore;
    }

    if (!best) { this.selectedId = null; return null; }
    this.selectedId = best.id;
    this.selectedBox = best.boxAt(now);     // 給下一輪的幾何繼承用

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
