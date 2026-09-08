// =============================================
// 前車起步判定
// =============================================
// 核心量測改成「影像尺度變化率」，而不是 v6 的逐點徑向投票。理由：
//
//   自車停著時，ego-motion 的擴張焦點（FOE）在數學上不存在
//   —— 沒有前進運動就沒有 FOE。v6 卻把 FOE 寫死在 (0.5W, 0.45H) 並據此
//   做徑向投影，是把一個不成立的幾何模型套上去；手機安裝角度一變、
//   前車不在畫面正中，徑向方向就有系統性偏斜，任何持續存在的未補償位移
//   都會投影出固定符號的票。配上「只打折不歸零」的 LLR，
//   穩態值是 net/(1−0.85) = net×6.7 —— 這是一個放著不動也會誤報的結構。
//
// 正確的物理量：車輛沿光軸遠離時，影像等比例縮小，且
//
//     d/dt log(w) = −(1/Z)·dZ/dt = −V_rel/Z = −1/TTC
//
//   所以定義 V ≡ −d·log(尺度)/dt，單位 1/秒，物理意義就是 1/TTC。
//   V > 0 ⇔ 正在遠離。
//
// 為什麼遠勝逐點投票：
//   * 手機抖動主要是平移與旋轉，幾乎不產生尺度變化
//     → 尺度這個自由度天然免疫抖動
//   * 80 個高度相關的點濃縮成 1 個參數 + 明確的標準誤
//     → SPRT 的「觀測獨立」假設終於成立（每 tick 一筆觀測，而非每點一筆）
//   * 不需要 FOE，不需要深度模型
//
// 兩個獨立判據必須同時成立才觸發：
//   (A) 卡爾曼濾波後的 V 顯著大於 0：z_kf = V/σ_V ≥ z_fire（由 α 推導）
//   (B) tick 級 SPRT 的累積 LLR ≥ A（由 α、β 推導）
// 再加上：物理最小值（1/TTC ≥ minInvTtc）、持續時間（dwell）、
//         影像往地平線方向移動的佐證、自車必須靜止。

import { clamp } from '../util/math.js';
import { CONFIG, derivedThresholds } from '../config.js';

export class DepartureDetector {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.th = derivedThresholds(cfg);
    this.reset();
  }

  reset() {
    // 一階隨機遊走卡爾曼：狀態就是 V（1/TTC），單位 1/秒
    this.V = 0;
    this.Vvar = 1e4;          // 初始極不確定
    this.inited = false;

    this.llr = 0;             // tick 級 SPRT 累積對數似然比
    this.ticks = 0;           // 有效量測 tick 數
    this.aboveSinceTs = 0;    // z_kf 連續達標的起始時刻
    this.upBuf = [];          // 近期 dyRel 的符號（佐證用）
    this.lastMeasTs = 0;
    // 衰減的參考時刻。coast() 每幀都會被呼叫（光流在累積基線時、
    // 重新錨定時、點數不足時都算沒有量測），若每次都拿
    // 「距上一筆量測的時間」當衰減量，同一段時間會被重複扣，
    // 證據被超線性地打掉 —— 這正是「該報的時候剛好沒報」的機制之一。
    this.lastDecayTs = 0;
    // -Infinity 而不是 0：否則啟動後的前 cooldownMs 毫秒內冷卻檢查永遠不通過
    this.lastFireTs = -Infinity;
    this.lastReason = 'idle';
    this.lastZ = 0;
    this.lastZRaw = 0;
    // 剎車燈熄滅的先驗只能加一次，不能每個 tick 重複加 ——
    // 重複計入同一份證據正是 v6 SPRT 的原始錯誤，不能在這裡犯第二次
    this.priorApplied = false;
  }

  /** 目標徹底遺失（超過 coast 時間）才呼叫；短暫遺失請用 coast() */
  onTargetLost() { this.reset(); }

  /** 本 tick 沒有可用量測（重新錨定、點數不足、背景不可信…）
   *  關鍵：不歸零證據，只讓它隨時間衰減。
   *  v6 在追蹤斷掉時 ofReleaseAll() 直接清空 LLR，
   *  而前車起步時 bbox 縮小恰好最容易讓追蹤斷掉 —— 這是漏報的直接機制。 */
  coast(ts) {
    const d = this.cfg.departure;
    if (this.lastMeasTs && ts - this.lastMeasTs > d.coastMs) {
      this.reset();
      this.lastReason = 'lost';
      return;
    }
    // 只扣「上次衰減到現在」這一小段，不是「距上一筆量測」的全長
    const dt = this.lastDecayTs ? (ts - this.lastDecayTs) / 1000 : 0;
    this.lastDecayTs = ts;
    if (dt > 0) {
      this.Vvar += d.qV * dt;                       // 不確定度隨時間膨脹
      this.llr *= Math.exp(-dt / d.llrDecayTau);    // 證據隨時間衰減
    }
    // 注意：不清掉 aboveSinceTs。dwell 要衡量的是「z 有沒有持續達標」，
    // 而光流在累積基線的那幾幀本來就沒有新量測，
    // 若每幀都把 dwell 歸零，dwell 這道閘門永遠不可能通過。
  }

  /**
   * 餵入一筆光流量測。
   * @param m  OpticalFlow.measure() 的成功回傳
   * @param ctx { ts, egoStill, trusted, primed, priorLlr, armed }
   *   primed   剎車燈剛熄滅 → 起步的先驗機率大幅提高
   *   priorLlr 該先驗的對數勝算比 ln(P(熄燈│起步)/P(熄燈│不起步))
   *   armed    目標已被連續追蹤足夠久（掠過的機車/切入車不該武裝）
   * @returns { fired: bool, status: {...} }
   */
  update(m, ctx) {
    const d = this.cfg.departure;
    const ts = ctx.ts;
    const primed = !!ctx.primed;

    if (!ctx.egoStill) {
      this.reset();
      this.lastReason = 'ego-moving';
      return { fired: false, status: this.status(ts) };
    }
    if (!ctx.trusted) {
      this.coast(ts);
      this.lastReason = 'untrusted';
      return { fired: false, status: this.status(ts) };
    }

    // ---- 本 tick 的原始標準化量測 ----
    // z_raw = −log(s_rel) / σ_log(s)。dt 在此自然消掉：
    // 較長的基線給出較大的 |log s| 但 σ 不變（σ 來自 LK 的像素雜訊），
    // 所以較長的基線本來就該得到較高的 SNR —— 這是正確的行為。
    const sigmaLog = m.sigmaRel / Math.max(Math.abs(m.sRel), 1e-6);
    if (!(sigmaLog > 0) || !isFinite(sigmaLog)) {
      this.coast(ts);
      this.lastReason = 'sigma-bad';
      return { fired: false, status: this.status(ts) };
    }
    const zRawUnclamped = -m.logSRel / sigmaLog;
    // 夾限：防止單一離群 tick 獨力衝過門檻（模型誤差不是高斯的）
    const zRaw = clamp(zRawUnclamped, -d.zClamp, d.zClamp);
    this.lastZRaw = zRawUnclamped;

    // ---- (A) 卡爾曼濾波 V（1/TTC）----
    const vMeas = -m.logSRel / m.dt;                       // 1/秒
    const vVarMeas = Math.pow(sigmaLog / m.dt, 2);
    if (!this.inited) {
      this.V = vMeas; this.Vvar = vVarMeas; this.inited = true;
    } else {
      const dtSec = this.lastDecayTs ? (ts - this.lastDecayTs) / 1000 : m.dt;
      this.Vvar += d.qV * Math.max(dtSec, 0);
      const K = this.Vvar / (this.Vvar + vVarMeas);
      this.V += K * (vMeas - this.V);
      this.Vvar *= (1 - K);
    }
    const sigmaV = Math.sqrt(Math.max(this.Vvar, 1e-12));
    const zKf = this.V / sigmaV;
    this.lastZ = zKf;

    // ---- (B) tick 級 SPRT（高斯平均值位移的精確 LLR）----
    //   H0: z ~ N(0, 1)      （靜止，純雜訊）
    //   H1: z ~ N(mu, 1)     （起步）
    //   單筆 LLR = mu·z − mu²/2
    const mu = d.effectSize;
    const dtDecay = this.lastDecayTs ? (ts - this.lastDecayTs) / 1000 : m.dt;
    this.llr *= Math.exp(-Math.max(dtDecay, 0) / d.llrDecayTau);
    this.llr += mu * zRaw - (mu * mu) / 2;

    // ---- 先驗：剎車燈熄滅是一個獨立來源的證據 ----
    // SPRT 累積的就是對數勝算比，所以先驗機率的改變剛好是「加一個常數」。
    // 這在數學上是乾淨的貝氏更新，而且結構上不可能單獨觸發：
    // 鬆了剎車但車不動 → 運動證據永遠不來 → LLR 停在先驗那一格然後被衰減掉。
    if (primed && !this.priorApplied) {
      this.llr += ctx.priorLlr || 0;
      this.priorApplied = true;
    }
    if (!primed) this.priorApplied = false;

    if (this.llr <= this.th.sprtB) this.llr = 0;           // 確認非起步 → 歸零重算

    // ---- 佐證：影像上前車應同時往地平線方向移動（dyRel < 0）----
    // 只有位移量超過雜訊尺度的量測才投票：次像素雜訊的符號是擲硬幣，
    // 讓它進投票池只會把比例鎖在 0.5 附近，等於一道隨機閘門。
    if (Math.abs(m.dyRel) >= d.upwardMinPx) {
      this.upBuf.push(m.dyRel <= 0 ? 1 : 0);
      if (this.upBuf.length > d.upwardWindow) this.upBuf.shift();
    }
    const upRatio = this.upBuf.length >= 3
      ? this.upBuf.reduce((a, b) => a + b, 0) / this.upBuf.length
      : 0;

    this.ticks++;
    this.lastMeasTs = ts;
    this.lastDecayTs = ts;

    // ---- 觸發條件 ----
    const condKf = zKf >= this.th.zFire;
    const condSprt = this.llr >= this.th.sprtA;
    const condPhysical = this.V >= d.minInvTtc;
    const condUp = !d.requireUpwardMotion || upRatio >= d.upwardAgreeRatio;

    if (condKf) {
      if (!this.aboveSinceTs) this.aboveSinceTs = ts;
    } else {
      this.aboveSinceTs = 0;
    }
    // 有獨立證據時放寬自我佐證的量（dwell 與 minTicks 的存在理由是
    // 「防止模型誤差的瞬時尖峰獨力觸發」，一個獨立來源正當地降低了這個需求）
    const needDwell = primed ? d.dwellMsPrimed : d.dwellMs;
    const needTicks = primed ? d.minTicksPrimed : d.minTicks;
    const dwellOk = this.aboveSinceTs > 0 && (ts - this.aboveSinceTs) >= needDwell;
    const ticksOk = this.ticks >= needTicks;
    const cooldownOk = ts - this.lastFireTs > d.cooldownMs;
    const armedOk = ctx.armed !== false;

    const statTest = d.requireBoth ? (condKf && condSprt) : (condKf || condSprt);
    const fired = statTest && condPhysical && condUp && dwellOk && ticksOk
      && cooldownOk && armedOk;

    this.lastReason = fired ? 'FIRE'
      : !armedOk ? 'not-armed'
      : !condPhysical ? 'below-min-ttc'
      : !condKf ? 'z-low'
      : !condSprt ? 'llr-low'
      : !condUp ? 'no-upward'
      : !dwellOk ? 'dwell'
      : !ticksOk ? 'few-ticks'
      : !cooldownOk ? 'cooldown'
      : '?';

    if (fired) {
      this.lastFireTs = ts;
      // 觸發後歸零統計證據，但保留 KF（車還在遠離，不該假裝沒看到）
      this.llr = 0;
      this.ticks = 0;
      this.aboveSinceTs = 0;
      this.upBuf.length = 0;
      this.priorApplied = false;
    }

    return { fired, status: this.status(ts, { upRatio, zRaw, condKf, condSprt, condPhysical, condUp, dwellOk, primed }) };
  }

  status(ts, extra = {}) {
    const ttc = this.V > 1e-4 ? 1 / this.V : Infinity;
    return {
      V: this.V,
      sigmaV: Math.sqrt(Math.max(this.Vvar, 0)),
      z: this.lastZ,
      zRaw: this.lastZRaw,
      llr: this.llr,
      ticks: this.ticks,
      ttc,
      reason: this.lastReason,
      zFire: this.th.zFire,
      sprtA: this.th.sprtA,
      dwellMs: this.aboveSinceTs ? ts - this.aboveSinceTs : 0,
      ...extra,
    };
  }

  /** 一行式除錯字串 */
  debugLine() {
    const s = this.status(performance.now());
    const ttc = isFinite(s.ttc) ? s.ttc.toFixed(1) + 's' : '--';
    return `V=${s.V.toFixed(3)}±${s.sigmaV.toFixed(3)} z=${s.z.toFixed(2)}/${s.zFire.toFixed(2)}`
      + ` LLR=${s.llr.toFixed(2)}/${s.sprtA.toFixed(2)} TTC=${ttc} n=${s.ticks} ${s.reason}`;
  }
}
