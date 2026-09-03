// =============================================
// bbox 卡爾曼濾波
// =============================================
// 為什麼不用 EMA：
//   v6 用 EMA_ALPHA = 0.18 平滑 bbox。EMA 假設固定取樣率，而 v6 的實際
//   tick 率會隨推論負載劇烈變動；更致命的是 α=0.18 代表約 5 個 tick 的
//   相位滯後 —— 前車起步時 bbox 快速縮小，平滑後的大框與真實小框 IoU
//   迅速掉到門檻以下 → 追蹤斷掉 → 證據被清空 → 就在最該報警的那一刻漏報。
//
// KF 的三個關鍵好處：
//   1. 明確用 dt 建模，幀率抖動不產生相位誤差
//   2. 有速度狀態，可以「預測到現在」——偵測結果延遲 200ms 回來也能對齊
//   3. 目標暫時被遮擋時可以慣性滑行（coast），不必立刻放棄
//
// 尺寸用 log 空間：log(w) 的線性變化對應「等比例縮放」，
// 這正是遠離／接近在影像上的物理行為，也讓雜訊更接近同質。

import { Kf1d } from '../util/math.js';
import { CONFIG } from '../config.js';

export class KalmanBox {
  constructor(box, ts, cfg = CONFIG) {
    const t = cfg.tracker;
    this.cx = new Kf1d({ q: t.qCenter, r: t.rCenter });
    this.cy = new Kf1d({ q: t.qCenter, r: t.rCenter });
    this.lw = new Kf1d({ q: t.qLogSize, r: t.rLogSize });
    this.lh = new Kf1d({ q: t.qLogSize, r: t.rLogSize });
    this.ts = ts;
    this._set(box);
  }

  _set(box) {
    this.cx.update(box.x + box.w / 2);
    this.cy.update(box.y + box.h / 2);
    this.lw.update(Math.log(Math.max(box.w, 1)));
    this.lh.update(Math.log(Math.max(box.h, 1)));
  }

  /** 用量測更新到時刻 ts */
  update(box, ts) {
    const dt = (ts - this.ts) / 1000;
    if (dt > 0) {
      this.cx.predict(dt); this.cy.predict(dt);
      this.lw.predict(dt); this.lh.predict(dt);
    }
    this.ts = ts;
    this._set(box);
  }

  /** 非破壞性外推到時刻 t（給關聯、ROI 定位、繪製用） */
  boxAt(t) {
    const dt = (t - this.ts) / 1000;
    const cx = this.cx.x + this.cx.v * dt;
    const cy = this.cy.x + this.cy.v * dt;
    const w = Math.exp(this.lw.x + this.lw.v * dt);
    const h = Math.exp(this.lh.x + this.lh.v * dt);
    return { x: cx - w / 2, y: cy - h / 2, w, h };
  }

  /** 影像尺度的變化率 d·log(w)/dt（1/秒）。負值 = 正在縮小 */
  get logScaleRate() {
    return (this.lw.v + this.lh.v) / 2;
  }

  /** 中心的垂直速度（px/s）。負值 = 影像上往地平線方向移動 */
  get vy() { return this.cy.v; }
}
