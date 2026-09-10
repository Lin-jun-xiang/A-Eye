// =============================================
// 多目標追蹤（SORT-lite）
// =============================================
// v6 的做法是「對單一 smoothedBbox 做貪婪 IoU 比對」，兩個後果：
//   1. 車流中 ID switch 很常見 → 鎖到旁車道的車起步 → 誤判
//   2. 前車起步時 bbox 縮小，IoU 掉破門檻 → 追蹤斷掉 → 漏報
// 這裡改成：KF 預測框做關聯 + 三重門（IoU / 中心距 / 尺寸比），
// 任一門通過即可關聯。IoU 掉到 0 時中心距這一路還能救回來。

import { KalmanBox } from './kalmanBox.js';
import { iou } from '../util/math.js';
import { CONFIG } from '../config.js';

let nextId = 1;

class Track {
  constructor(det, ts, cfg) {
    this.id = nextId++;
    this.kf = new KalmanBox(det, ts, cfg);
    this.classId = det.classId;
    this.score = det.score;
    this.hits = 1;
    this.misses = 0;
    this.lastBox = { x: det.x, y: det.y, w: det.w, h: det.h };
    this.lastBoxTs = ts;
    this.firstTs = ts;
    this.lastSeenTs = ts;
    this.confirmed = false;
  }
  boxAt(t) { return this.kf.boxAt(t); }
  get ageMs() { return this.lastSeenTs - this.firstTs; }
}

export class Tracker {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.tracks = [];
    // 關聯的收支帳。實車除錯需要分辨兩種完全不同的失效：
    //   偵測進來但配不上 → 關聯門檻的問題
    //   偵測根本沒進來   → 偵測器的問題
    // 沒有這兩個數字，只看得到「目標一直是舊的」而不知道該修哪一邊。
    this.stats = { dets: 0, detsLow: 0, matched: 0, recoveredLow: 0, created: 0, dropped: 0 };
  }

  reset() {
    this.tracks.length = 0;
    this.stats = { dets: 0, detsLow: 0, matched: 0, recoveredLow: 0, created: 0, dropped: 0 };
  }

  /**
   * @param dets 偵測結果 [{x,y,w,h,score,classId}]
   * @param ts   這些偵測所對應的影像時間戳（不是現在！）
   */
  update(dets, ts) {
    const t = this.cfg.tracker;
    const tracks = this.tracks;
    const confHigh = this.cfg.yolo.confThreshold;

    // ---- BYTE：把偵測分成高分層與低分層 ----
    // 低分框通常是**真的物體**（被遮擋、動態模糊、夜間過曝），不是背景。
    // 實車量測：那台前車的偵測信心中位數只有 0.19，門檻 0.30 → 78% 的時間
    // 它根本進不了這個函式。但直接降門檻會讓護欄與反光一起變成車，
    // 所以低分框只走第二段、而且只能延續**既有**的 track ——
    // 「有沒有對應的既有軌跡」就是分辨真物體與背景的先驗。
    const hi = [], lo = [];
    for (const d of dets) (d.score >= confHigh ? hi : lo).push(d);

    const tUsed = new Uint8Array(tracks.length);

    // ---- 第一段：高分框 × 全部 track（IoU / 中心距 / 尺寸比 三重門）----
    const matchedHi = this._assign(tracks, hi, ts, tUsed, false);

    // ---- 第二段：低分框 × 「第一段沒配到的」track，只用 IoU、門檻更緊 ----
    // 這一段不建立新 track，沒配上的低分框直接丟掉。
    const matchedLo = this._assign(tracks, lo, ts, tUsed, true);

    // ---- 未配對的 track：累積 miss（但先不刪，允許 coast）----
    for (let i = 0; i < tracks.length; i++) if (!tUsed[i]) tracks[i].misses++;

    // ---- 新 track 只從**高分**框建立 ----
    let created = 0;
    for (const d of hi) {
      if (!d._used) { tracks.push(new Track(d, ts, this.cfg)); created++; }
    }
    for (const d of dets) delete d._used;

    // ---- 淘汰超過 coast 時間的 track ----
    const before = tracks.length;
    this.tracks = tracks.filter((tr) => ts - tr.lastSeenTs <= t.maxCoastMs);

    this.stats.dets += hi.length;
    this.stats.detsLow += lo.length;
    this.stats.matched += matchedHi;
    this.stats.recoveredLow += matchedLo;
    this.stats.created += created;
    this.stats.dropped += before - this.tracks.length;
    return this.tracks;
  }

  /**
   * 貪婪指派一批偵測到尚未配對的 track。
   * @param lowTier true → BYTE 的第二段：只用 IoU、門檻更緊、且**不算入
   *                confirmHits**（幽靈 track 不該只靠低分框就被確認成前車）
   * @returns 成功配對的數量
   */
  _assign(tracks, dets, ts, tUsed, lowTier) {
    if (!dets.length) return 0;
    const t = this.cfg.tracker;
    const pairs = [];
    for (let i = 0; i < tracks.length; i++) {
      if (tUsed[i]) continue;
      const pred = tracks[i].boxAt(ts);
      const diag = Math.hypot(pred.w, pred.h);
      for (let j = 0; j < dets.length; j++) {
        const d = dets[j];
        if (d.classId !== tracks[i].classId && !this._sameGroup(d.classId, tracks[i].classId)) continue;

        const ov = iou(pred, d);
        const dc = Math.hypot(
          (pred.x + pred.w / 2) - (d.x + d.w / 2),
          (pred.y + pred.h / 2) - (d.y + d.h / 2)
        );
        const areaRatio = (d.w * d.h) / Math.max(pred.w * pred.h, 1);

        // 尺寸差太多一定不是同一台（這是硬性幾何約束，不是感覺）
        if (areaRatio > t.scaleGate || areaRatio < 1 / t.scaleGate) continue;

        if (lowTier) {
          // 第二段刻意不給中心距那條寬鬆的救援路徑：低分框本來就比較可能
          // 是背景，唯一該救回來的是「明顯就落在預測框上」的那些。
          if (ov < t.iouGateLow) continue;
        } else {
          const passIou = ov >= t.iouGate;
          const passCenter = diag > 0 && dc <= diag * t.centerGateRatio;
          if (!passIou && !passCenter) continue;
        }

        // 成本：以 IoU 為主，中心距為輔（IoU=0 時仍有梯度可用）
        const cost = (1 - ov) + 0.5 * (diag > 0 ? dc / diag : 1);
        pairs.push({ i, j, cost });
      }
    }

    // 貪婪全域指派（成本升序）。N 很小（<20），沒必要上匈牙利演算法
    pairs.sort((a, b) => a.cost - b.cost);
    const dUsed = new Uint8Array(dets.length);
    let matched = 0;
    for (const p of pairs) {
      if (tUsed[p.i] || dUsed[p.j]) continue;
      tUsed[p.i] = 1; dUsed[p.j] = 1;
      const tr = tracks[p.i], d = dets[p.j];
      tr.kf.update(d, ts);
      tr.score = d.score;
      tr.classId = d.classId;
      // 「這個 track 這一刻的原始量測框」——bbox 尺度變化率那條路要用它，
      // 而且必須是量測、不是 KF 外推（外推值跨時間高度相關）。
      tr.lastBox = { x: d.x, y: d.y, w: d.w, h: d.h };
      tr.lastBoxTs = ts;
      tr.misses = 0;
      tr.lastSeenTs = ts;
      // 低分框只延續軌跡，不累積「確認」用的命中數
      if (!lowTier) {
        tr.hits++;
        if (tr.hits >= this.cfg.tracker.confirmHits) tr.confirmed = true;
      }
      d._used = true;
      matched++;
    }
    return matched;
  }

  /** car / bus / truck 之間允許類別跳動（YOLO 對大車常在這三類間搖擺） */
  _sameGroup(a, b) {
    const veh = this.cfg.vehicleClasses;
    return veh.includes(a) && veh.includes(b);
  }

  byId(id) { return this.tracks.find((t) => t.id === id) || null; }

  confirmedOf(classIds, now, maxCoastMs = this.cfg.tracker.maxCoastMs) {
    return this.tracks.filter(
      (t) => t.confirmed && classIds.includes(t.classId) && now - t.lastSeenTs <= maxCoastMs
    );
  }
}
