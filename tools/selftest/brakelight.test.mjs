// 剎車燈判定的行為測試
//
// 用合成 RGBA 影像驅動 lampStats() + BrakeLightDetector.updateFromStats()，
// 兩者都不接觸 DOM，所以可以在 node 裡跑。要驗證的是「四種干擾都不會誤報」：
//   方向燈（單側）、雙閃（週期性）、紅色車身（漆面不是燈）、逆光過曝（必須回 unknown）
// 以及夜間情境：尾燈長亮時，剎車燈仍然要能靠「相對自身峰值」分辨出來。

import { lampStats, BrakeLightDetector } from '../../src/logic/brakeLight.js';
import { CONFIG } from '../../src/config.js';

const B = CONFIG.brakeLight;
const W = 120, H = 80;

/**
 * 合成一張「車尾」：車身底色 + 左右兩顆燈。
 * @param body  [r,g,b] 車身顏色
 * @param lampL [r,g,b] 左燈顏色
 * @param lampR [r,g,b] 右燈顏色
 */
function render(body, lampL, lampR) {
  const d = new Uint8ClampedArray(W * H * 4);
  const put = (x, y, c) => {
    const i = (y * W + x) * 4;
    d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) put(x, y, body);
  // 燈的位置放在 sideFrac 區內、yTop~yBottom 帶內
  const y0 = Math.round(H * 0.45), y1 = Math.round(H * 0.75);
  for (let y = y0; y < y1; y++) {
    for (let x = Math.round(W * 0.04); x < Math.round(W * 0.25); x++) put(x, y, lampL);
    for (let x = Math.round(W * 0.75); x < Math.round(W * 0.96); x++) put(x, y, lampR);
  }
  return d;
}

const GREY = [60, 60, 60];
const BRAKE = [235, 25, 25];      // 剎車燈：飽和紅
const TAIL = [95, 22, 22];        // 尾燈：同一燈室、亮度低得多
const DARK = [55, 45, 45];        // 燈熄（白天）
const AMBER = [235, 155, 25];     // 方向燈：琥珀（G 高 → 紅度低）
const WHITE = [252, 252, 252];    // 過曝

const feed = (det, data, t) => det.updateFromStats(lampStats(data, W, H, B), t);

let fails = 0;
const check = (label, cond, extra = '') => {
  console.log(`  ${cond ? '✅' : '✗ '} ${label}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

console.log('=== 基本：亮 → 熄，且熄滅需持續 offConfirmMs ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const on = render(GREY, BRAKE, BRAKE);
  const off = render(GREY, DARK, DARK);
  let t = 0;
  for (; t < 2000; t += 100) feed(det, on, t);
  // 只看過一個位準時，無法知道那是尾燈還是剎車燈 —— 所以只能說「有紅燈亮著」。
  // 判「熄」有證據（落差本身就是證據），判「剎車中」不能用預設值充當。
  check('只看過一個位準 → state=lit（有紅燈但分不出尾燈/剎車）',
    det.state === 'lit', det.lastDetail);

  const r1 = feed(det, off, t); t += 100;
  check('熄滅第 1 幀還不承認（需持續 ' + B.offConfirmMs + 'ms）', !r1.released && det.state !== 'off');

  let released = false, releaseAt = 0;
  const t0 = t - 100;
  for (; t < t0 + 1200; t += 100) {
    const r = feed(det, off, t);
    if (r.released) { released = true; releaseAt = t - t0; }
  }
  check('持續熄滅後觸發 released', released, `延遲 ${releaseAt}ms`);
  check('released 只發生一次（邊緣事件）',
    !feed(det, off, t + 100).released);

  // 有了「亮」與「熄」兩個位準，動態範圍才算解析 → 之後再踩下去就敢說剎車中
  t += 200;
  for (let i = 0; i < 10; i++, t += 100) feed(det, on, t);
  check('經歷一次熄滅後，再踩剎車 → state=on（範圍已解析）',
    det.state === 'on', det.lastDetail);
}

console.log('');
console.log('=== 夜間：尾燈長亮，剎車燈靠「相對自身峰值」分辨 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const tail = render(GREY, TAIL, TAIL);
  const brake = render(GREY, BRAKE, BRAKE);
  let t = 0;
  for (; t < 1500; t += 100) feed(det, tail, t);          // 只有尾燈
  for (; t < 4000; t += 100) feed(det, brake, t);         // 踩剎車
  check('踩剎車 → on', det.state === 'on', det.lastDetail);
  let released = false;
  for (; t < 6000; t += 100) if (feed(det, tail, t).released) released = true;
  check('鬆剎車（回到尾燈亮度）→ released', released, det.lastDetail);
}

console.log('');
console.log('=== 干擾 1：方向燈（單側）不得判成鬆剎車 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const on = render(GREY, BRAKE, BRAKE);
  let t = 0;
  for (; t < 2000; t += 100) feed(det, on, t);
  // 右側改成琥珀方向燈、左側維持剎車燈亮
  const turn = render(GREY, BRAKE, AMBER);
  let released = false;
  for (; t < 5000; t += 100) if (feed(det, turn, t).released) released = true;
  check('單側變化 → 不觸發', !released, det.lastDetail);
}

console.log('');
console.log('=== 干擾 2：雙閃（對稱但週期性）不得判成鬆剎車 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const on = render(GREY, BRAKE, BRAKE);
  const off = render(GREY, DARK, DARK);
  let t = 0;
  for (; t < 2000; t += 100) feed(det, on, t);
  let releases = 0;
  // 真實雙閃約 1.2~1.5Hz（熄滅相 ~330ms）本來就短於 offConfirmMs=350ms
  // → 第一道防線就攔掉了。這裡刻意用「熄滅相 700ms」的慢速閃爍，
  //   讓每個循環都足以觸發，才真正測到閃爍抑制那段邏輯。
  for (let cyc = 0; cyc < 5; cyc++) {
    for (let i = 0; i < 7; i++, t += 100) if (feed(det, off, t).released) releases++;
    for (let i = 0; i < 4; i++, t += 100) if (feed(det, on, t).released) releases++;
  }
  check('慢速週期性閃爍：只放過第 1 次，之後判為 BLINK 抑制',
    releases <= 1 && det.blinking, `releases=${releases} blinking=${det.blinking}`);

  // 第一道防線：真實雙閃頻率下，熄滅相根本撐不到 offConfirmMs
  const det2 = new BrakeLightDetector(CONFIG);
  let t2 = 0, rel2 = 0;
  for (; t2 < 2000; t2 += 100) feed(det2, on, t2);
  for (let cyc = 0; cyc < 6; cyc++) {                      // 1.5Hz：熄 330ms / 亮 330ms
    for (let i = 0; i < 3; i++, t2 += 110) if (feed(det2, off, t2).released) rel2++;
    for (let i = 0; i < 3; i++, t2 += 110) if (feed(det2, on, t2).released) rel2++;
  }
  check('1.5Hz 雙閃：熄滅相短於 offConfirmMs → 一次都不觸發', rel2 === 0, `releases=${rel2}`);
}

console.log('');
console.log('=== 干擾 3：紅色車身（漆面不是燈）不得判成有燈 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const redCar = render([185, 45, 45], [190, 48, 48], [190, 48, 48]);   // 全紅、燈與車身同色
  let t = 0;
  let released = false;
  for (; t < 4000; t += 100) if (feed(det, redCar, t).released) released = true;
  const st = lampStats(redCar, W, H, B);
  check('紅色車身不算「有一對紅燈」', !det.everOn,
    `contrast=${(Math.min(st.left, st.right) / (st.body + 6)).toFixed(2)} (門檻 ${B.minContrast})`);
  check('因此也不會觸發鬆剎車', !released);
}

console.log('');
console.log('=== 干擾 4：逆光過曝必須回報 unknown，不能回報「熄滅」 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const on = render(GREY, BRAKE, BRAKE);
  let t = 0;
  for (; t < 2000; t += 100) feed(det, on, t);
  const blown = render(WHITE, WHITE, WHITE);
  let released = false, sawUnknown = false, usable = true;
  for (; t < 5000; t += 100) {
    const r = feed(det, blown, t);
    if (r.released) released = true;
    if (r.state === 'unknown') sawUnknown = true;
    if (r.usable === false) usable = false;
  }
  check('過曝 → state=unknown', sawUnknown);
  check('過曝 → 絕不觸發鬆剎車（否則夕陽直射會誤報）', !released);
  check('過曝 → usable=false（不得計入「沒看到剎車燈」的證據）', !usable);
}

console.log('');
console.log('=== 實車夜間量測的回歸測試（數字取自路測影片，非合成）===');
// 2026-09-08 夜間路測，前車 SUV 停等紅燈後起步。用修好的量測腳本在真實畫格上
// 量到的數值（左/右燈）：
//   剎車燈亮：燈芯亮度 137/120、面積 0.43/0.30、對比 1.80~1.94
//   剎車燈熄：燈芯亮度  88/ 86、面積 0.12/0.11、對比 3.0~6.9
// 注意兩件事，這是整個設計的關鍵證據：
//   1. 亮度只掉到 0.64/0.72 —— 單看亮度永遠達不到 offRatio=0.45（原本的 bug）
//   2. 對比在熄燈時反而「更高」—— 因為亮燈時光暈把中央參考區也照紅了，
//      所以 contrast 只能用來判斷「有一對紅燈」，不能用來判斷「剎車燈亮」
{
  // 直接組出對應的直方圖：比例 a 的像素在紅度 v，其餘在背景紅度 bg
  const mkHist = (a, v, bg, n = 4000) => {
    const h = new Int32Array(32);
    h[Math.min(31, (v / 8) | 0)] = Math.round(n * a);
    h[Math.min(31, (bg / 8) | 0)] = n - Math.round(n * a);
    return h;
  };
  const mk = (coreL, coreR, aL, aR, body) => ({
    left: coreL, right: coreR, body,
    histL: mkHist(aL, coreL, 8), histR: mkHist(aR, coreR, 8),
    nL: 4000, nR: 4000,
    overexposed: 0.02, luma: 62, n: 12000,
  });
  const LIT = mk(137, 120, 0.43, 0.30, 61);      // contrast = 120/(61+6) = 1.79
  const DARK = mk(88, 86, 0.12, 0.11, 12);       // contrast = 86/(12+6) = 4.8

  const det = new BrakeLightDetector(CONFIG);
  let t = 0;
  for (; t < 3000; t += 100) det.updateFromStats(LIT, t);
  check('亮燈時 state=lit（尚未看過落差，不敢斷言是剎車）',
    det.state === 'lit', det.lastDetail);
  const lit = det.peakL;

  let released = false, at = 0;
  const tOff = t;
  for (; t < tOff + 1500; t += 100) {
    const r = det.updateFromStats(DARK, t);
    if (r.released && !released) { released = true; at = t - tOff; }
  }
  check('熄燈後觸發 released', released, `延遲 ${at}ms（offConfirmMs=${B.offConfirmMs}）`);
  const ratio = (0.12 * 88) / lit;
  console.log(`     位準比 熄/亮 = ${ratio.toFixed(2)}（門檻 ${B.offRatio}）`
    + `；若只看亮度則是 ${(88 / 137).toFixed(2)} → 永遠不會觸發`);
  check('只看亮度的話會漏掉（證明必須用面積×亮度）', 88 / 137 > B.offRatio);
}

console.log('');
console.log('=== 先驗：熄燈給 SPRT 的對數勝算比 ===');
{
  const det = new BrakeLightDetector(CONFIG);
  const llr = det.priorLlr();
  const expect = Math.log(B.pOffGivenDepart / B.pOffGivenStay);
  console.log(`  ln(${B.pOffGivenDepart}/${B.pOffGivenStay}) = ${llr.toFixed(2)}`
    + `（SPRT 上門檻 A = ${Math.log((1 - CONFIG.departure.beta) / CONFIG.departure.alpha).toFixed(2)}）`);
  check('先驗值與兩個機率一致', Math.abs(llr - expect) < 1e-9);
  check('先驗單獨不足以觸發（結構上不可能靠鬆剎車就報）',
    llr < Math.log((1 - CONFIG.departure.beta) / CONFIG.departure.alpha),
    `${llr.toFixed(2)} < 門檻`);
}

console.log('');
if (fails) {
  console.log(`❌ ${fails} 項未通過`);
  process.exitCode = 1;
} else {
  console.log('✅ 全部通過');
}
