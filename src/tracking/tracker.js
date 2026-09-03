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
  }

  reset() { this.tracks.length = 0; }

  /**
   * @param dets 偵測結果 [{x,y,w,h,score,classId}]
   * @param ts   這些偵測所對應的影像時間戳（不是現在！）
   */
  update(dets, ts) {
    const t = this.cfg.tracker;
    const tracks = this.tracks;

    // 1) 建立所有「可行」的 (track, det) 配對及其成本
    const pairs = [];
    for (let i = 0; i < tracks.length; i++) {
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

        const passIou = ov >= t.iouGate;
        const passCenter = diag > 0 && dc <= diag * t.centerGateRatio;
        if (!passIou && !passCenter) continue;

        // 成本：以 IoU 為主，中心距為輔（IoU=0 時仍有梯度可用）
        const cost = (1 - ov) + 0.5 * (diag > 0 ? dc / diag : 1);
        pairs.push({ i, j, cost });
      }
    }

    // 2) 貪婪全域指派（成本升序）。N 很小（<20），沒必要上匈牙利演算法
    pairs.sort((a, b) => a.cost - b.cost);
    const tUsed = new Uint8Array(tracks.length);
    const dUsed = new Uint8Array(dets.length);
    for (const p of pairs) {
      if (tUsed[p.i] || dUsed[p.j]) continue;
      tUsed[p.i] = 1; dUsed[p.j] = 1;
      const tr = tracks[p.i], d = dets[p.j];
      tr.kf.update(d, ts);
      tr.score = d.score;
      tr.classId = d.classId;
      tr.hits++;
      tr.misses = 0;
      tr.lastSeenTs = ts;
      if (tr.hits >= this.cfg.tracker.confirmHits) tr.confirmed = true;
    }

    // 3) 未配對的 track：累積 miss（但先不刪，允許 coast）
    for (let i = 0; i < tracks.length; i++) if (!tUsed[i]) tracks[i].misses++;

    // 4) 未配對的偵測 → 新 track
    for (let j = 0; j < dets.length; j++) {
      if (!dUsed[j]) tracks.push(new Track(dets[j], ts, this.cfg));
    }

    // 5) 淘汰超過 coast 時間的 track
    this.tracks = tracks.filter((tr) => ts - tr.lastSeenTs <= t.maxCoastMs);
    return this.tracks;
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
