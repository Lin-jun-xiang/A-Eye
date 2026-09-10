// 起步判定器的蒙地卡羅驗證
// 目的：在有 ground truth 的合成訊號上量出「誤警率 / 偵測延遲」，
//       確認 config 裡的 α 真的對應到實際誤警率（v6 的 α 完全不對應）。

import { DepartureDetector } from '../../src/motion/departure.js';
import { CONFIG } from '../../src/config.js';

// 從實測估的 LK 雜訊尺度：
//   σ_resid ≈ 0.3px, rRms ≈ 60px（ROI 384、車佔 200px）, n ≈ 50
//   σ_s = σ_resid / (rRms·√n) ≈ 0.3 / (60·7.07) ≈ 7.1e-4
const SIGMA_LOG = 7.1e-4;
const HZ = 20;
const DT = 1 / HZ;

let seed = 12345;
function rand() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function gauss() {
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeMeas(logS, dyRel) {
  const sRel = Math.exp(logS);
  return { ok: true, dt: DT, sRel, logSRel: logS, sigmaRel: SIGMA_LOG * sRel, dyRel };
}

/** @returns { fires, firstFireTick } */
function run(nTicks, vTrue, biasSigma, dySign) {
  const det = new DepartureDetector(CONFIG);
  let fires = 0, first = -1;
  for (let i = 0; i < nTicks; i++) {
    const ts = i * DT * 1000;
    // 真實的 log 尺度變化 = -V·dt；再加上量測雜訊與系統性偏壓
    const logS = -vTrue * DT + SIGMA_LOG * (gauss() - biasSigma);
    const r = det.update(makeMeas(logS, dySign), { ts, egoStill: true, trusted: true });
    if (r.fired) { fires++; if (first < 0) first = i; }
  }
  return { fires, first };
}

const HOUR_TICKS = 3600 * HZ;
console.log(`模擬設定：${HZ} Hz、σ_log(s) = ${SIGMA_LOG.toExponential(2)}`);
console.log(`config: alpha=${CONFIG.departure.alpha}, minInvTtc=${CONFIG.departure.minInvTtc}/s, dwell=${CONFIG.departure.dwellMs}ms, minTicks=${CONFIG.departure.minTicks}`);
console.log('');

console.log('=== 誤警（前車完全靜止）===');
for (const bias of [0, 0.25, 0.5, 1.0]) {
  const { fires } = run(HOUR_TICKS, 0, bias, -1);
  console.log(`  系統性偏壓 ${bias.toFixed(2)}σ/tick → ${fires} 次誤警 / 小時`);
}
console.log('');

console.log('=== 偵測延遲（前車確實起步；假設 10m 前方）===');
for (const [v, desc] of [[0.02, '0.2 m/s 極慢'], [0.05, '0.5 m/s 緩起步'], [0.10, '1.0 m/s'], [0.20, '2.0 m/s'], [0.40, '4.0 m/s 急起步']]) {
  const { first } = run(HZ * 10, v, 0, -1);
  const lat = first < 0 ? '未偵測' : `${(first * DT * 1000).toFixed(0)} ms`;
  console.log(`  V=${v.toFixed(2)}/s (TTC ${(1 / v).toFixed(0)}s, ${desc}) → ${lat}`);
}
console.log('');

console.log('=== 反向情境（應永不觸發）===');
{
  const { fires } = run(HOUR_TICKS, -0.10, 0, 1);
  console.log(`  前車正在靠近 (V=-0.10/s) → ${fires} 次觸發`);
}
{
  // 起步了但影像往下移（佐證不符，例如追到的其實是路面陰影）
  const { fires } = run(HZ * 30, 0.20, 0, +1);
  console.log(`  尺度縮小但影像往下移（佐證不符）→ ${fires} 次觸發`);
}
{
  const det = new DepartureDetector(CONFIG);
  let fires = 0;
  for (let i = 0; i < HZ * 30; i++) {
    const ts = i * DT * 1000;
    const logS = -0.3 * DT + SIGMA_LOG * gauss();
    if (det.update(makeMeas(logS, -1), { ts, egoStill: false, trusted: true }).fired) fires++;
  }
  console.log(`  自車行駛中（egoStill=false）→ ${fires} 次觸發`);
}
console.log('');

console.log('=== 追蹤中斷的韌性（v6 在此漏報）===');
{
  // 起步中途，模擬 400ms 追蹤斷掉（coast），看是否還能報出來
  const det = new DepartureDetector(CONFIG);
  let first = -1;
  for (let i = 0; i < HZ * 10; i++) {
    const ts = i * DT * 1000;
    const inGap = i >= 5 && i < 5 + Math.round(0.4 * HZ);
    if (inGap) { det.coast(ts); continue; }
    const logS = -0.20 * DT + SIGMA_LOG * gauss();
    if (det.update(makeMeas(logS, -1), { ts, egoStill: true, trusted: true }).fired && first < 0) first = i;
  }
  console.log(`  V=0.20/s 且中途斷追 400ms → ${first < 0 ? '未偵測' : (first * DT * 1000).toFixed(0) + ' ms'}`);
}
{
  // 徹底遺失超過 coastMs → 應歸零重來
  const det = new DepartureDetector(CONFIG);
  for (let i = 0; i < 5; i++) det.update(makeMeas(-0.2 * DT, -1), { ts: i * 50, egoStill: true, trusted: true });
  const llrBefore = det.llr;
  det.coast(5 * 50 + CONFIG.departure.coastMs + 100);
  console.log(`  遺失 > coastMs(${CONFIG.departure.coastMs}ms) → LLR ${llrBefore.toFixed(2)} 歸零為 ${det.llr.toFixed(2)}，reason=${det.lastReason}`);
}

console.log('');
console.log('=== 真實起步（等加速度 2 m/s^2，不同車距）===');
for (const Z of [5, 10, 20, 30]) {
  const det = new DepartureDetector(CONFIG);
  let first = -1;
  for (let i = 0; i < HZ * 6; i++) {
    const ts = i * DT * 1000;
    const t = i * DT;
    const v = 2.0 * t;                 // m/s
    const z = Z + 0.5 * 2.0 * t * t;   // 目前距離
    const V = v / z;                   // 1/TTC
    const logS = -V * DT + SIGMA_LOG * gauss();
    if (det.update(makeMeas(logS, -1), { ts, egoStill: true, trusted: true }).fired && first < 0) first = i;
  }
  console.log(`  ${Z}m 前方、2 m/s^2 起步 → ${first < 0 ? '未偵測' : (first * DT * 1000).toFixed(0) + ' ms（此時車速 ' + (2 * first * DT).toFixed(2) + ' m/s）'}`);
}

console.log('');
console.log('=== 模型誤差突波（v6 的主要誤警型態）===');
for (const [nBad, magSigma] of [[1, 20], [1, 100], [3, 20], [5, 10], [10, 5]]) {
  const det = new DepartureDetector(CONFIG);
  let fires = 0;
  for (let i = 0; i < HZ * 20; i++) {
    const ts = i * DT * 1000;
    // 在第 40 tick 起插入 nBad 個「一致同向」的離群量測（相關雜訊，非高斯）
    const bad = i >= 40 && i < 40 + nBad;
    const logS = bad ? -SIGMA_LOG * magSigma : SIGMA_LOG * gauss();
    if (det.update(makeMeas(logS, -1), { ts, egoStill: true, trusted: true }).fired) fires++;
  }
  console.log(`  ${nBad} 個連續 ${magSigma}σ 的一致離群 tick → ${fires} 次誤警`);
}

console.log('');
console.log('=== coast() 的衰減不得重複計算同一段時間 ===');
// 光流不是每一幀都產生量測（累積基線中、重新錨定、點數不足都算沒量測），
// 所以 coast() 會以視訊幀率被呼叫。若衰減量用「距上一筆量測的時間」，
// 同一段時間會被扣好幾次，證據被超線性打掉 —— 該報的時候剛好報不出來。
{
  const tau = CONFIG.departure.llrDecayTau;
  const build = () => {
    const det = new DepartureDetector(CONFIG);
    for (let i = 0; i < 6; i++) {
      det.update(makeMeas(-SIGMA_LOG * 3, -1), { ts: i * 50, egoStill: true, trusted: true });
    }
    return det;
  };
  const gapMs = 400;                      // 空白 400ms，理論衰減 exp(-0.4/tau)
  const a = build(), b = build();
  const llr0 = a.llr;
  a.coast(250 + gapMs);                   // 一次呼叫
  for (let i = 1; i <= 12; i++) b.coast(250 + (gapMs / 12) * i);   // 每 33ms 呼叫一次
  const expect = llr0 * Math.exp(-gapMs / 1000 / tau);
  console.log(`  LLR ${llr0.toFixed(2)} 空白 ${gapMs}ms 後：`
    + `理論 ${expect.toFixed(3)}｜呼叫 1 次 ${a.llr.toFixed(3)}｜呼叫 12 次 ${b.llr.toFixed(3)}`);
  const ok = Math.abs(a.llr - b.llr) < 1e-9 && Math.abs(a.llr - expect) < 1e-6;
  console.log(`  → ${ok ? '✅ 與呼叫次數無關' : '✗ 衰減量取決於呼叫次數'}`);
}

console.log('');
console.log('=== ego 抖一下不得清空證據（最後一處硬歸零）===');
let egoFails = 0;
const check = (label, cond, note = '') => {
  if (!cond) egoFails++;
  console.log(`  ${cond ? '✅' : '✗ '} ${label}${note ? '  ' + note : ''}`);
};
// 自車狀態是逐 tick 判定的，會抖。原本 ego-moving 走 this.reset()，
// 於是一個雜訊 tick 就能把 7 筆量測、LLR=6.2 全部歸零（2026-09-10 實測）。
// 安全性靠的是 canAlert 擋住觸發，不是靠清空證據。
{
  const cfg = CONFIG;
  const det = new DepartureDetector(cfg);
  // armed:false → 證據照常累積但永遠不觸發，這樣測到的才是「證據有沒有被清掉」
  const ctx = (ts, egoStill) => ({ ts, egoStill, trusted: true, primed: false, priorLlr: 0, armed: false });
  // V=0.05（緩起步）：會累積證據但在這個時間尺度內還不會觸發，
  // 這樣才測得到「證據有沒有被清掉」而不是「有沒有觸發」
  const m = () => makeMeas(-0.05 * DT, -2);

  let t = 0;
  for (let i = 0; i < 10; i++, t += DT * 1000) det.update(m(), ctx(t, true));
  const before = { n: det.ticks, llr: det.llr };
  check(`累積到 n=${before.n} LLR=${before.llr.toFixed(1)}（尚未觸發）`,
    before.n >= 5 && before.llr > 1, `n=${before.n}`);

  det.update(m(), ctx(t, false)); t += DT * 1000;   // 一個 ego=moving 的雜訊 tick
  check('單一 ego-moving tick 後證據仍在（只衰減，不歸零）',
    det.ticks === before.n && det.llr > before.llr * 0.5,
    `n=${det.ticks} LLR=${det.llr.toFixed(2)}（原本 ${before.llr.toFixed(2)}）`);
  check('reason 有誠實回報', det.lastReason === 'ego-moving');

  // 但持續行駛超過 coastMs 就該歸零
  for (let i = 0; i < 40; i++, t += DT * 1000) det.update(m(), ctx(t, false));
  check(`持續行駛 ${cfg.departure.coastMs}ms 以上 → 證據歸零`,
    det.ticks === 0 && det.llr === 0, `n=${det.ticks} LLR=${det.llr.toFixed(2)}`);
}
if (egoFails) process.exitCode = 1;
