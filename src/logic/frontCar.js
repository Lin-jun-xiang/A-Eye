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

import { clamp } from '../util/math.js';

export class FrontCarSelector {
  constructor(cfg) {
    this.cfg = cfg;
    this.selectedId = null;
    // 車道中心的橫向位置（畫面比例）。手機常常沒有裝在正中央，
    // 所以用極慢的速率線上學習，並夾在 ±0.15 內防止跑掉。
    this.laneCenterX = 0.5;
    this.horizon = cfg.frontCar.horizonFallback;
    this.aspect = 9 / 16;       // vh/vw，每 tick 由實際影像尺寸更新
  }

  reset() { this.selectedId = null; }

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
    // (3) 走廊權重過低（注意：車道線模型只做軟性加權，不參與這個硬閘）
    if (this.corridorWeight(box, vw, vh) < f.minCorridorWeight) return false;
    return true;
  }

  /**
   * 從 track 清單中選出前車。
   * @param tracks  已確認的車輛 track
   * @param now     現在時刻（用於 KF 外推）
   * @param laneWeightFn (box) => 0~1 或 null，車道線的軟性加權
   * @returns 選中的 track 或 null
   */
  select(tracks, vw, vh, now, laneWeightFn = null) {
    let best = null, bestScore = -1;
    let current = null, currentScore = -1;

    for (const tr of tracks) {
      const box = tr.boxAt(now);
      if (!this.isCandidate(box, vw, vh)) continue;

      const cw = this.corridorWeight(box, vw, vh);
      const lw = laneWeightFn ? laneWeightFn(box) : null;
      const laneW = (lw === null || lw === undefined) ? 1 : (0.5 + 0.5 * lw);

      // 底邊越低 = 越近（透視幾何）。正規化到 0~1。
      const proximity = clamp(
        ((box.y + box.h) / vh - this.horizon) / Math.max(1 - this.horizon, 1e-3), 0, 1
      );
      const score = proximity * cw * laneW;

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
    return `horizon=${this.horizon.toFixed(2)} laneCx=${this.laneCenterX.toFixed(3)}`;
  }
}
