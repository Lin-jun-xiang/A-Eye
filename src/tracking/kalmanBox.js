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
    this.cfg = cfg;
    this.cx = new Kf1d({ q: t.qCenter, r: t.rCenter });
    this.cy = new Kf1d({ q: t.qCenter, r: t.rCenter });
    this.lw = new Kf1d({ q: t.qLogSize, r: t.rLogSize });
    this.lh = new Kf1d({ q: t.qLogSize, r: t.rLogSize });
    this.ts = ts;
    // ---- 長寬比先驗（log 空間），2026-09-11 手持實測逼出來的 ----
    //
    // 同一台靜止的前車，DETR 的框在「只框車尾」與「連儀表板一起框」兩種
    // 模式間跳：高度 370↔645（變異 22.6%），而寬度只有 0.78%。
    // 濾波後畫面上的框因此打氣筒式脹縮（實測 KF 高度 330~708、變異 23.1%），
    // 而剎車燈取樣區是「框的固定比例」—— 框一吞掉儀表板，取樣區就落在
    // 儀表板上，把亮著的燈讀成熄滅 → 誤觸「鬆開剎車」。
    //
    // 判別依據是剛體不變量，不是門檻猜測：**同一台車的寬與高必須等比例
    // 變化**（兩者都 ∝ 1/Z）。高度單獨跳、寬度不動 = 不是尺度變化，
    // 是遮擋邊界的分割抖動。此時以寬度為準（實測乾淨的那個維度）、
    // 用長寬比先驗重建高度，並**固定上緣**（上緣是乾淨邊：實測靜止時
    // 上緣抖 11px、下緣抖 270px —— 汙染幾乎都發生在下緣）。
    this.la = Math.log(Math.max(box.w, 1)) - Math.log(Math.max(box.h, 1));
    this.laSigma = cfg.tracker.aspectSigmaFloor;
    this._set(box);
  }

  /**
   * 長寬比守門：量測的長寬比偏離先驗太多 → 其中一個維度被汙染。
   *
   * 哪一個？**不寫死方向** —— 手持工況的汙染在高度（儀表板把下緣往下拖，
   * 2026-09-11 實測），但支架夜景的汙染是部分偵測（只框到半台車 →
   * 寬度被砍）。第一版寫死「信任寬度」，在支架素材上把部分偵測的窄寬度
   * 拿去重建高度，整個框跟著縮小 → 關聯變弱 → 證據清空 4 → 10 次。
   *
   * 對稱的規則：信任「與 KF 自身預測較一致」的維度（創新量較小者），
   * 用長寬比先驗重建另一個。重建高度時固定上緣（下緣才是遮擋汙染發生處）；
   * 重建寬度時固定中心 x（部分偵測砍的是哪一側無從得知）。
   */
  _gateAspect(box) {
    const t = this.cfg.tracker;
    const lwM = Math.log(Math.max(box.w, 1));
    const lhM = Math.log(Math.max(box.h, 1));
    const resid = (lwM - lhM) - this.la;
    const gated = Math.abs(resid) > t.aspectGate * Math.max(this.laSigma, t.aspectSigmaFloor);
    if (!gated) {
      // 正常樣本：更新先驗與其雜訊尺度（σ 由資料自己估，floor 只防 0）
      this.la += t.aspectEma * resid;
      this.laSigma = (1 - t.aspectEma) * this.laSigma + t.aspectEma * Math.abs(resid);
      return box;
    }
    // 被汙染的樣本：先驗仍以極慢速率跟隨 —— 否則第一筆就汙染的 track
    // 會把後面所有乾淨樣本都當成離群，先驗永遠修不回來（死鎖）。
    this.la += t.aspectEmaGated * resid;
    // this.lw / this.lh 在 update() 裡剛 predict 過 → .x 就是「預測到現在」
    const innovW = Math.abs(lwM - this.lw.x);
    const innovH = Math.abs(lhM - this.lh.x);
    if (innovW <= innovH) {
      // 寬度與軌跡一致 → 高度被汙染：重建高度、上緣固定
      const h = Math.max(box.w, 1) / Math.exp(this.la);
      return { x: box.x, y: box.y, w: box.w, h };
    }
    // 高度與軌跡一致 → 寬度被汙染（部分偵測）：重建寬度、中心 x 固定
    const w = Math.max(box.h, 1) * Math.exp(this.la);
    const cx = box.x + box.w / 2;
    return { x: cx - w / 2, y: box.y, w, h: box.h };
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
    this._set(this._gateAspect(box));
  }

  /**
   * 非破壞性外推到時刻 t（給關聯、ROI 定位、繪製用）。
   *
   * 外推用**阻尼速度**而不是線性：位移 = v·τ·(1−e^(−dt/τ))。
   * dt ≪ τ 時趨近 v·dt（快節拍下行為不變）；dt 大時位移有界（v·τ）。
   *
   * 為什麼必要：實機偵測間隔 0.8~1.5 秒（DETR 1.2Hz），線性外推等於
   * 讓上一筆量測的速度雜訊不受約束地跑 1 秒以上 —— 使用者看到的
   * 「框一直跑掉、越變越寬」就是 lw.v 恰為正時的線性外推。
   * 量測驅動的速度只在約一個偵測週期內可信，超過就該衰減。
   * 離線跑機（偵測準時、零延遲）重現不了這件事，這是實機才有的病。
   */
  boxAt(t) {
    const dt = (t - this.ts) / 1000;
    const tau = this.cfg.tracker.coastVelocityTauMs / 1000;
    // 分段：一個偵測週期內**維持線性** —— 下一筆偵測抵達時的關聯
    // 需要完整的運動預測（全程阻尼實測讓 8Hz 支架影片的紅綠燈小框
    // 漏配，綠燈事件晚了 1.6 秒、目標新鮮 88%→80%）。
    // 超過一個週期才開始阻尼：那之後的速度已無量測支持，是純雜訊外推。
    // grace 由 tracker 餵入實測偵測間隔；冷啟動（尚無量測）時用 τ 保底。
    const t0 = (this.coastGraceMs > 0 ? this.coastGraceMs : this.cfg.tracker.coastVelocityTauMs) / 1000;
    let eff;
    if (dt <= 0) eff = 0;
    else if (dt <= t0) eff = dt;
    else eff = t0 + tau * (1 - Math.exp(-(dt - t0) / tau));
    // 阻尼**只作用在尺寸**，中心維持線性外推。
    // 二分實測（乾淨.mp4 44~52s）：中心也阻尼會把綠燈事件拖慢 1.6 秒 ——
    // 紅綠燈的框在變燈瞬間會真的快速移動（DETR 框的是亮著的那顆燈，
    // 紅上綠下），coast 期間框被阻尼留在舊位置，取色一直取到熄掉的紅燈。
    // 而使用者抱怨的「越變越寬」是尺寸的病：靜止目標的尺寸速度是純雜訊，
    // 位置速度卻可能承載真實運動 —— 兩者的可信度不對稱，處置就該不對稱。
    const effC = Math.max(dt, 0);
    const cx = this.cx.x + this.cx.v * effC;
    const cy = this.cy.x + this.cy.v * effC;
    const w = Math.exp(this.lw.x + this.lw.v * eff);
    const h = Math.exp(this.lh.x + this.lh.v * eff);
    return { x: cx - w / 2, y: cy - h / 2, w, h };
  }

  /** 影像尺度的變化率 d·log(w)/dt（1/秒）。負值 = 正在縮小 */
  get logScaleRate() {
    return (this.lw.v + this.lh.v) / 2;
  }

  /** 中心的垂直速度（px/s）。負值 = 影像上往地平線方向移動 */
  get vy() { return this.cy.v; }
}
