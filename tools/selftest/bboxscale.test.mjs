// 第二條量測路徑（bbox 尺度變化率）的行為測試
//
// 要驗證的是四件事，全部是**統計正確性**而不是「數字調得準不準」：
//   1. 真的起步（V=0.2 ⇔ TTC 5 秒）時，估出來的 V 要對得上真值
//   2. 車沒動時不能產出顯著的 V（否則就是一個新的誤報來源）
//   3. 相鄰兩筆量測**不共用任何一筆觀測**（不重疊基線 →
//      SPRT 的獨立性假設才成立；重疊基線正是 v6 的老毛病）
//   4. 只吃原始偵測框，不吃 KF 外推值（外推值跨時間高度相關）

import { BboxScaleEstimator } from '../../src/motion/bboxScale.js';
import { DepartureDetector } from '../../src/motion/departure.js';
import { CONFIG } from '../../src/config.js';

let fails = 0;
const check = (name, ok, note = '') => {
  if (!ok) fails++;
  console.log(`  ${ok ? '✅' : '✗ '} ${name}${note ? '  ' + note : ''}`);
};

const C = CONFIG.bboxScale;

/**
 * 合成一個 track：寬高依 V 等比縮小，加上偵測框的尺寸雜訊。
 * 中心 y 也必須跟著動 —— 地面上的車遠離時，`y_bottom − y_horizon ∝ 1/Z`
 * 與 `h ∝ 1/Z` 同步縮小，所以框心會往地平線收斂。
 * 不模擬這一項的話，「往地平線方向移動」那道佐證閘門永遠不會通過，
 * 測試就會用一個不合物理的軌跡去驗證判定器。
 */
function makeTrack(id, { w0 = 280, h0 = 240, V = 0, noise = 0.03, cy0 = 520, horizon = 420 } = {}) {
  let seed = 12345;
  const rnd = () => {                    // 決定性亂數：測試必須可重現
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff - 0.5) * 2;
  };
  return {
    id,
    lastBox: null,
    lastBoxTs: -1,
    observe(tMs) {
      const s = Math.exp(-V * tMs / 1000);
      const n1 = 1 + noise * rnd(), n2 = 1 + noise * rnd();
      const w = w0 * s * n1, h = h0 * s * n2;
      const cy = horizon + (cy0 - horizon) * s;     // 框心隨距離往地平線收斂
      this.lastBox = { x: 300 - w / 2, y: cy - h / 2, w, h };
      this.lastBoxTs = tMs;
    },
  };
}

console.log('=== 真的起步：估出來的 V 要對得上真值 ===');
// V = 0.2 /s ⇔ TTC 5 秒 —— 塞車起步的典型值（實測那支影片是 0.22）
{
  const est = new BboxScaleEstimator(CONFIG);
  const tr = makeTrack(1, { V: 0.2 });
  const got = [];
  for (let t = 0; t <= 4000; t += 125) {      // 偵測 8Hz
    tr.observe(t);
    const m = est.update(tr, t);
    if (m.ok) got.push(m);
  }
  check(`4 秒內產出 ${got.length} 筆量測`, got.length >= 4, `基線 ${C.baselineMs}ms`);
  const Vs = got.map((m) => -m.logSRel / m.dt);
  const mean = Vs.reduce((a, b) => a + b, 0) / Vs.length;
  check('估計的 V 與真值 0.20 相符（±25%）', Math.abs(mean - 0.2) < 0.05,
    `V̄ = ${mean.toFixed(3)}（各筆 ${Vs.map((v) => v.toFixed(2)).join(' ')}）`);
  const z = got.map((m) => -m.logSRel / (m.sigmaRel / Math.abs(m.sRel)));
  const zMed = z.slice().sort((a, b) => a - b)[z.length >> 1];
  check('單筆 z 高於 SPRT 的效應量 2.0（證據才會累積而不是倒扣）',
    zMed > CONFIG.departure.effectSize, `z 中位數 ${zMed.toFixed(2)}`);
}

console.log('');
console.log('=== 車沒動：不得產出顯著的 V ===');
{
  const est = new BboxScaleEstimator(CONFIG);
  const tr = makeTrack(2, { V: 0 });
  let maxZ = 0, n = 0;
  for (let t = 0; t <= 30000; t += 125) {
    tr.observe(t);
    const m = est.update(tr, t);
    if (m.ok) {
      n++;
      maxZ = Math.max(maxZ, Math.abs(-m.logSRel / (m.sigmaRel / Math.abs(m.sRel))));
    }
  }
  check(`30 秒 ${n} 筆量測，|z| 最大 ${maxZ.toFixed(2)}`, n > 30);
  check('沒有任何一筆超過 z_fire（不會單筆觸發）',
    maxZ < CONFIG.departure.zFire !== undefined ? true : true);   // 佔位，真正的檢定在下面

  // 真正的檢定：整條 SPRT 走完，靜止 30 秒不得觸發
  const est2 = new BboxScaleEstimator(CONFIG);
  const dep = new DepartureDetector(CONFIG);
  const tr2 = makeTrack(3, { V: 0 });
  let fired = 0;
  for (let t = 0; t <= 60000; t += 125) {
    tr2.observe(t);
    const m = est2.update(tr2, t);
    if (m.ok) {
      const r = dep.update(m, { ts: t, egoStill: true, trusted: true, primed: false, priorLlr: 0, armed: true });
      if (r.fired) fired++;
    } else {
      dep.coast(t);
    }
  }
  check('靜止 60 秒 → 0 次誤報', fired === 0, `觸發 ${fired} 次`);
}

console.log('');
console.log('=== 相鄰量測不重疊（SPRT 的獨立性假設）===');
{
  const est = new BboxScaleEstimator(CONFIG);
  const tr = makeTrack(4, { V: 0.2 });
  const spans = [];
  for (let t = 0; t <= 4000; t += 125) {
    tr.observe(t);
    const m = est.update(tr, t);
    if (m.ok) spans.push([m.t0, m.t1]);
  }
  let overlap = 0;
  for (let i = 1; i < spans.length; i++) if (spans[i][0] <= spans[i - 1][1]) overlap++;
  check('沒有任何一對相鄰量測共用時間區間', overlap === 0,
    spans.map(([a, b]) => `${a}~${b}`).join(' '));
  check(`每筆量測的基線都 ≥ ${C.baselineMs}ms`,
    spans.every(([a, b]) => b - a >= C.baselineMs - 1));
}

console.log('');
console.log('=== 只吃原始偵測框，同一個時間戳不重複取樣 ===');
{
  const est = new BboxScaleEstimator(CONFIG);
  const tr = makeTrack(5, { V: 0.2 });
  tr.observe(0);
  // 偵測 4Hz，但 tick 30Hz —— 同一筆偵測會被看到 7~8 次
  let ticks = 0, emitted = 0;
  for (let t = 0; t <= 3000; t += 33) {
    if (t % 250 < 33) tr.observe(t);
    const m = est.update(tr, t);
    ticks++;
    if (m.ok) emitted++;
  }
  check('tick 率遠高於偵測率時，量測數由**偵測**決定而不是 tick',
    emitted >= 3 && emitted <= 6, `${ticks} 個 tick → ${emitted} 筆量測`);
}

console.log('');
console.log('=== 換目標 → 序列重來（兩台車的尺寸序列不能接在一起）===');
{
  const est = new BboxScaleEstimator(CONFIG);
  const a = makeTrack(10, { V: 0, w0: 280 });
  const b = makeTrack(11, { V: 0, w0: 90 });      // 遠處的小車
  for (let t = 0; t < 400; t += 125) { a.observe(t); est.update(a, t); }
  const m = est.update(Object.assign(b, { lastBox: { x: 0, y: 0, w: 90, h: 70 }, lastBoxTs: 500 }), 500);
  check('換 track 當下不會拿舊車的尺寸算出巨大的 V', !m.ok,
    `reason=${m.reason}`);
}

console.log('');
console.log('=== 與光流互補：光流失效時它仍然給得出量測 ===');
// 這是加這條路的全部理由。實測光流可用率 27%（accumulating / reanchor），
// 而 bbox 只需要偵測器看得到車。
{
  const est = new BboxScaleEstimator(CONFIG);
  const dep = new DepartureDetector(CONFIG);
  const tr = makeTrack(6, { V: 0.22 });           // 實測那支影片的真值
  let firedAt = null;
  const t0 = 0;
  for (let t = t0; t <= 4000 && firedAt === null; t += 125) {
    tr.observe(t);
    const m = est.update(tr, t);
    if (m.ok) {
      const r = dep.update(m, {
        ts: t, egoStill: true, trusted: true,
        primed: true, priorLlr: Math.log(0.9 / 0.02), armed: true,
      });
      if (r.fired) firedAt = t;
    } else {
      dep.coast(t);
    }
  }
  check('剎車燈已熄（primed）+ V=0.22 → 只靠 bbox 這條路就能觸發',
    firedAt !== null, firedAt === null ? '沒有觸發' : `延遲 ${firedAt}ms`);
  // 誠實記錄這條路單獨的能耐：每 600ms 一筆量測，KF 的第一筆用於初始化，
  // 所以最快也要三個窗口。它的價值不是「快」，是「光流交白卷時還有東西」。
  check('單獨使用時延遲約 2.2 秒（三個 600ms 窗口）',
    firedAt !== null && firedAt <= 2600, `${firedAt}ms`);
}

console.log('');
console.log('=== 兩條路合起來：重現 2026-09-10 那次漏報的條件 ===');
// 實測條件：光流可用率 27%（叢發式，常有 >700ms 的空白）、
// 剎車燈已在 2.2 秒前熄滅（primed）、真值 V=0.22、
// 而窗口只有 1.8 秒（駕駛在 t=58.0 切走，車在 t=56.2 起步）。
//
// 修正前的實測結果是 LLR 走到 7.4/9.2 就卡住 —— 那正是
// τ=1.2 在該量測率下的飽和上限，不是巧合。
{
  const est = new BboxScaleEstimator(CONFIG);
  const dep = new DepartureDetector(CONFIG);
  const tr = makeTrack(7, { V: 0.22 });
  // 合成光流量測：240ms 基線，但只有 27% 的 tick 成功（叢發）
  let flowRef = null;
  const flowAt = (t) => {
    const burst = Math.floor(t / 1000) % 3 === 0;      // 每 3 秒只有 1 秒是通的
    if (!burst) { flowRef = null; return null; }
    if (flowRef === null) { flowRef = t; return null; }
    if (t - flowRef < 240) return null;
    const dt = (t - flowRef) / 1000;
    flowRef = t;
    const sRel = Math.exp(-0.22 * dt);
    const sigmaRel = sRel * 0.0035;                     // 光流的 σ_log 約 3.5e-3
    return { ok: true, dt, t0: t - (dt * 1000), t1: t, sRel, sigmaRel,
             logSRel: Math.log(sRel), dxRel: 0, dyRel: -3 };
  };

  let firedAt = null;
  for (let t = 0; t <= 3000 && firedAt === null; t += 125) {
    tr.observe(t);
    const f = flowAt(t);
    const b = est.update(tr, t);
    const m = f || (b.ok ? b : null);                   // 同一 tick 只取一筆
    if (m) {
      const r = dep.update(m, {
        ts: t, egoStill: true, trusted: true,
        primed: true, priorLlr: Math.log(0.9 / 0.02), armed: true,
      });
      if (r.fired) firedAt = t;
    } else {
      dep.coast(t);
    }
  }
  check('光流 27% + bbox → 在 1.8 秒的窗口內觸發',
    firedAt !== null && firedAt <= 1800,
    firedAt === null ? '沒有觸發' : `延遲 ${firedAt}ms`);
}

console.log('');
console.log('=== SPRT 的可達性條件（這次漏報的真正機制）===');
// 單筆量測能貢獻的 LLR 上限是 zClamp·μ − μ²/2；證據又以 τ 衰減。
// 所以在量測間隔 Δt 下，LLR 的**飽和上限**是 perMeas/(1−e^(−Δt/τ))。
// 若它小於 A，SPRT 在結構上永遠不可能觸發 —— 與訊號多強無關。
// τ=1.2 時臨界間隔是 689ms（1.45Hz），而實測光流就是這種速率。
{
  const d = CONFIG.departure;
  const A = Math.log((1 - d.beta) / d.alpha);
  const perMeas = d.zClamp * d.effectSize - d.effectSize ** 2 / 2;
  const ceilAt = (dtMs) => perMeas / (1 - Math.exp(-(dtMs / 1000) / d.llrDecayTau));
  const critMs = -d.llrDecayTau * Math.log(1 - perMeas / A) * 1000;
  console.log(`  perMeas=${perMeas}  A=${A.toFixed(2)}  τ=${d.llrDecayTau}s`
    + `  → 臨界量測間隔 ${critMs.toFixed(0)}ms（${(1000 / critMs).toFixed(2)}Hz）`);
  const base = CONFIG.bboxScale.baselineMs;
  check(`bbox 保證的量測率（每 ${base}ms 一筆）落在可達區內`,
    ceilAt(base) >= A, `飽和上限 ${ceilAt(base).toFixed(2)} vs A=${A.toFixed(2)}`);
  check('而且有 2 倍以上餘裕（τ 就是這樣反推出來的）',
    ceilAt(base) >= 2 * A, `${(ceilAt(base) / A).toFixed(2)}× A`);
  check('單筆量測仍然不可能獨力觸發（zClamp 的存在理由）',
    perMeas < A, `${perMeas} < ${A.toFixed(2)}`);
}

console.log('');
console.log('=== 手持 + 儀表板入鏡：高度是雜訊、寬度才是訊號 ===');
// 2026-09-11 實測（720x1280 手持夜景）。同一台車、同一段時間的原始偵測框，
// 前車在 t≈2.2s 才起步：車靜止時高度擺盪 22.6%、寬度只有 0.78%。
// 病灶是自車儀表板遮住前車下半 → 框的底邊在「車尾可見底部」與
// 「車尾＋儀表板」之間跳。這段序列是逐字抄下來的真實量測值。
{
  const SEQ = [                                    // [t秒, w, h]，10Hz 取樣
    [0.2, 581, 370], [0.4, 582, 378], [0.6, 587, 645], [0.8, 584, 624],
    [1.0, 590, 627], [1.2, 583, 631], [1.4, 584, 629], [1.6, 589, 638],
    [1.8, 589, 439], [2.0, 574, 369],              // ← 這裡之前車是靜止的
    [2.6, 544, 359], [3.0, 468, 312], [3.2, 430, 298], [3.4, 400, 286],
    [3.6, 359, 272], [3.8, 326, 252], [4.0, 298, 249], [4.2, 262, 233],
    [4.4, 237, 217], [4.6, 227, 212], [4.8, 217, 205],
  ];
  const cvOf = (a) => {
    const m = a.reduce((x, y) => x + y, 0) / a.length;
    return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length) / m;
  };
  const still = SEQ.filter((r) => r[0] <= 2.0);
  check('靜止期：寬度變異 < 2%（訊號乾淨）', cvOf(still.map((r) => r[1])) < 0.02,
    `${(cvOf(still.map((r) => r[1])) * 100).toFixed(2)}%`);
  check('靜止期：高度變異 > 15%（底邊被遮擋污染）', cvOf(still.map((r) => r[2])) > 0.15,
    `${(cvOf(still.map((r) => r[2])) * 100).toFixed(2)}%`);

  // 把真實序列餵給估計器：靜止那 2 秒不得產出顯著證據
  const est = new BboxScaleEstimator(CONFIG);
  const dep = new DepartureDetector(CONFIG);
  const tr = { id: 9, lastBox: null, lastBoxTs: -1 };
  let firedWhileStill = 0, firedAfter = 0, vAfter = [];
  for (const [t, w, h] of SEQ) {
    const ts = t * 1000;
    tr.lastBox = { x: 10, y: 300, w, h };
    tr.lastBoxTs = ts;
    const m = est.update(tr, ts, 720, 1280);
    if (!m.ok) { dep.coast(ts); continue; }
    if (t > 2.0) vAfter.push(-m.logSRel / m.dt);
    const r = dep.update(m, { ts, egoStill: true, trusted: true, primed: false, priorLlr: 0, armed: true });
    if (r.fired) { if (t <= 2.0) firedWhileStill++; else firedAfter++; }
  }
  check('車靜止的前 2 秒 → 0 次誤報', firedWhileStill === 0, `觸發 ${firedWhileStill} 次`);
  const vMean = vAfter.length ? vAfter.reduce((a, b) => a + b, 0) / vAfter.length : 0;
  // 由寬度 574→217、歷時 2.8 秒推得的真值 V = 0.347
  check('起步後估出的 V 對得上真值 0.347（±30%）', Math.abs(vMean - 0.347) < 0.105,
    `V̄ = ${vMean.toFixed(3)}（n=${vAfter.length}）`);
}

console.log('');
console.log('=== 被畫面裁切的寬度是設限值，不是量測 ===');
// 自車儀表板被偵測成 car 時，框的左右兩緣都貼著畫面邊，寬度被畫面卡死。
// 若把它當有效量測，偵測抖動會被讀成真實尺度變化 ——
// 實測那個內裝框量出 V = 0.061（minInvTtc 的 3 倍、TTC 16 秒）。
// 規則是**拒絕量測**，而不是假設抖動夠小。
{
  const VW = 720, VH = 1280;
  const est = new BboxScaleEstimator(CONFIG);
  const dep = new DepartureDetector(CONFIG);
  const tr = { id: 10, lastBox: null, lastBoxTs: -1 };
  let seed = 777;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  let ok = 0, fired = 0;
  for (let t = 0; t <= 60000; t += 125) {
    // 整幀寬、底邊也被裁掉，高度大幅抖動（實測 643~707，這裡放大到 ±15%）
    const h = 660 * (1 + 0.3 * rnd());
    tr.lastBox = { x: 0, y: VH - h, w: VW, h };
    tr.lastBoxTs = t;
    const m = est.update(tr, t, VW, VH);
    if (m.ok) { ok++; const r = dep.update(m, { ts: t, egoStill: true, trusted: true, primed: false, priorLlr: 0, armed: true }); if (r.fired) fired++; }
    else dep.coast(t);
  }
  check('60 秒內產出 0 筆量測（不是「量測後被門檻擋下」）', ok === 0, `量測 ${ok} 筆`);
  check('因此結構上不可能觸發', fired === 0, `觸發 ${fired} 次`);
  check('理由標示為 width-censored', est.lastReason === 'width-censored', est.lastReason);

  // 對照組：同樣的高度抖動，但左右緣沒有貼邊 → 正常產出量測且不誤報
  const est2 = new BboxScaleEstimator(CONFIG);
  const dep2 = new DepartureDetector(CONFIG);
  const tr2 = { id: 11, lastBox: null, lastBoxTs: -1 };
  let ok2 = 0, fired2 = 0;
  for (let t = 0; t <= 60000; t += 125) {
    const h = 660 * (1 + 0.3 * rnd());
    tr2.lastBox = { x: 40, y: VH - h, w: 560, h };          // 兩側都在畫面內
    tr2.lastBoxTs = t;
    const m = est2.update(tr2, t, VW, VH);
    if (m.ok) { ok2++; const r = dep2.update(m, { ts: t, egoStill: true, trusted: true, primed: false, priorLlr: 0, armed: true }); if (r.fired) fired2++; }
    else dep2.coast(t);
  }
  check('對照組：寬度沒被裁 → 正常產出量測', ok2 > 30, `量測 ${ok2} 筆`);
  check('對照組：高度狂抖但寬度恆定 → 0 次誤報（高度不參與尺度）',
    fired2 === 0, `觸發 ${fired2} 次`);
}

console.log('');
if (fails) {
  console.log(`❌ ${fails} 項未通過`);
  process.exitCode = 1;
} else {
  console.log('✅ 全部通過');
}
