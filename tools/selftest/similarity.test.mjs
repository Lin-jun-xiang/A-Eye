// 相似變換擬合的行為測試
//
// 這個估計量原本委託給 cv.estimateAffinePartial2D，而**官方 opencv.js 根本沒有
// 編入那個函式**（opencv/opencv#20538）→ 每次呼叫都丟例外 → 尺度變化率這條路
// 從 v7 寫出來就沒運作過一次。實車路測的面板證實了：flow ok 0/225 fg-fit-fail:41
//
// 改成純 JS 的閉式解之後，它終於可以被離線測試。重點不只是「解對不對」，
// 而是**它報出來的 σ_s 誠不誠實** —— 整個 SPRT 的門檻都建立在這個標準誤上，
// 如果 σ_s 低報，誤警率就會遠高於設定的 α。

import { fitSimilarity } from '../../src/motion/opticalFlow.js';
import { CONFIG } from '../../src/config.js';

const F = CONFIG.flow;
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const gauss = () => {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
let fails = 0;
const check = (label, cond, extra = '') => {
  console.log(`  ${cond ? '✅' : '✗ '} ${label}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

/** 產生 n 個點，套上已知的相似變換（+ 雜訊 + 離群點） */
function makePair(n, s, theta, tx, ty, noise = 0, outFrac = 0) {
  const prev = new Float32Array(n * 2), next = new Float32Array(n * 2);
  // 注意角度慣例：擬合回報的是 θ = atan2(−b, a)，所以這裡用 b = −s·sinθ
  // 才能讓「產生的角度」與「回報的角度」直接比較
  const a = s * Math.cos(theta), b = -s * Math.sin(theta);
  for (let i = 0; i < n; i++) {
    const x = 40 + rnd() * 200, y = 40 + rnd() * 200;
    prev[i * 2] = x; prev[i * 2 + 1] = y;
    let ex = a * x + b * y + tx, ey = -b * x + a * y + ty;
    if (outFrac > 0 && rnd() < outFrac) { ex += (rnd() - 0.5) * 40; ey += (rnd() - 0.5) * 40; }
    else { ex += gauss() * noise; ey += gauss() * noise; }
    next[i * 2] = ex; next[i * 2 + 1] = ey;
  }
  return { prev, next };
}

console.log('=== 無雜訊時應精確還原 ===');
{
  const { prev, next } = makePair(40, 0.95, 0.03, 3, -2);
  const f = fitSimilarity(prev, next, F);
  console.log(`  s=${f.s.toFixed(6)}（真值 0.95）  θ=${f.theta.toFixed(6)}（真值 0.03）`
    + `  t=(${f.tx.toFixed(3)}, ${f.ty.toFixed(3)})（真值 3, -2）`);
  check('尺度誤差 < 1e-5', Math.abs(f.s - 0.95) < 1e-5);
  check('旋轉誤差 < 1e-5', Math.abs(f.theta - 0.03) < 1e-5);
  check('平移誤差 < 1e-3', Math.hypot(f.tx - 3, f.ty + 2) < 1e-3);
}

console.log('');
console.log('=== 回報的 σ_s 必須與實際散布一致（SPRT 的門檻靠它）===');
for (const [n, noise] of [[20, 0.3], [40, 0.3], [80, 0.3], [40, 0.8]]) {
  const trials = 400;
  let sum = 0, sum2 = 0, sigSum = 0;
  for (let k = 0; k < trials; k++) {
    const { prev, next } = makePair(n, 1.0, 0, 0, 0, noise);
    const f = fitSimilarity(prev, next, F);
    sum += f.s; sum2 += f.s * f.s; sigSum += f.sigmaS;
  }
  const mean = sum / trials;
  const emp = Math.sqrt(Math.max(sum2 / trials - mean * mean, 0));
  const rep = sigSum / trials;
  const ratio = rep / emp;
  console.log(`  n=${String(n).padStart(2)} 雜訊 ${noise}px → 實際散布 σ=${emp.toExponential(2)}`
    + `  回報 σ_s=${rep.toExponential(2)}  比值 ${ratio.toFixed(2)}`);
  check(`  n=${n}/${noise}px：回報的 σ_s 沒有低報（比值 ≥ 0.7）`, ratio >= 0.7);
  check(`  n=${n}/${noise}px：也沒有過度保守（比值 ≤ 2.0）`, ratio <= 2.0);
}

console.log('');
console.log('=== 20% 離群點下仍要還原尺度（MAD 剔除）===');
{
  let worst = 0;
  for (let k = 0; k < 100; k++) {
    const { prev, next } = makePair(60, 0.97, 0.01, 2, 1, 0.3, 0.20);
    const f = fitSimilarity(prev, next, F);
    worst = Math.max(worst, Math.abs(f.s - 0.97));
  }
  console.log(`  100 次試驗中最差的尺度誤差 = ${worst.toExponential(2)}（真值 0.97）`);
  check('最差誤差 < 0.01（1% 尺度 ≈ 1/TTC 誤差 0.04/s @240ms 基線）', worst < 0.01);
}

console.log('');
console.log('=== 退化與異常輸入 ===');
{
  check('點數不足回 null', fitSimilarity(new Float32Array(6), new Float32Array(6), F) === null);
  const same = new Float32Array([10, 10, 10, 10, 10, 10, 10, 10]);
  check('所有點重合（den=0）回 null', fitSimilarity(same, same, F) === null);
  const { prev, next } = makePair(30, 3.0, 0, 0, 0);      // 尺度 3 倍 → 離譜
  check('尺度離譜（3×）視為擬合失敗', fitSimilarity(prev, next, F) === null);
  // 共線點：實車量測顯示點雲並不共線（λ1/λ2≈1.4），但仍要確認不會爆
  const cn = 20;
  const cp = new Float32Array(cn * 2), cq = new Float32Array(cn * 2);
  for (let i = 0; i < cn; i++) {
    cp[i * 2] = 20 + i * 10; cp[i * 2 + 1] = 100;
    cq[i * 2] = (20 + i * 10) * 0.98; cq[i * 2 + 1] = 100 * 0.98;
  }
  const f = fitSimilarity(cp, cq, F);
  check('共線點不會回 NaN', f !== null && isFinite(f.s), f ? `s=${f.s.toFixed(4)}` : '');
}

console.log('');
console.log(fails ? `❌ ${fails} 項未通過` : '✅ 全部通過');
if (fails) process.exitCode = 1;
