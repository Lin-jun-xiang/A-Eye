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
console.log('=== 第三剎車燈（車頂中央那一顆）===');
// 為什麼它比外側燈好一個數量級：外側那兩顆兼作尾燈，踩剎車只是「變更亮」
// （實測 122 → 65，只差 1.5 倍）；第三剎車燈不接尾燈電路，只在踩剎車時亮
// （實測 145 → 0，差 70 倍）。
//
// 下面的數字全部取自 2026-09-08 夜間路測的真實網格：
//   踩著：燈格 142~147、網格中位數 23~39
//   鬆開：燈格 0、網格中位數 5~7
{
  const C = B.chmsl;
  /** 直接組出 lampStats 的輸出（含第三剎車燈的搜尋網格） */
  const mkSt = ({ ch, chBg, side = 60, body = 20 }) => {
    const vals = new Float32Array(C.cols * C.rows).fill(chBg);
    // 燈在中央上部：實測是 12 格中的第 4~5 列、6 行中的第 0~1 行
    for (const rx of [4, 5]) for (const ry of [0, 1]) vals[ry * C.cols + rx] = ch;
    const h = new Int32Array(32);
    h[Math.min(31, (side / 8) | 0)] = 1200;
    h[1] = 2800;
    return {
      left: side, right: side, body,
      histL: h, histR: h, nL: 4000, nR: 4000,
      chGrid: {
        vals, cols: C.cols, rows: C.rows,
        cellW: C.searchXFrac / C.cols,
        cellH: C.searchYFrac / C.rows,
        xOff: 0.5 - C.searchXFrac / 2,
      },
      overexposed: 0.02, luma: 62, n: 12000,
    };
  };
  const LIT = mkSt({ ch: 145, chBg: 23 });
  const DARK = mkSt({ ch: 0, chBg: 6 });

  const det = new BrakeLightDetector(CONFIG);
  let t = 0, pressAt = null, relAt = null;
  // 先暗一段（駕駛沒踩剎車 —— 實測那位駕駛整個停等 38 秒都沒踩）
  for (; t < 1500; t += 100) det.updateFromStats(DARK, t, 270);
  check('沒踩剎車時不會誤判為亮', det.chState !== 'on', det.lastDetail.split('\n')[1]);

  const tOn = t;
  for (; t < tOn + 1500; t += 100) {
    const r = det.updateFromStats(LIT, t, 270);
    if (r.pressed && pressAt === null) pressAt = t - tOn;
  }
  check('踩下剎車 → 偵測到「踩下」', pressAt !== null, `延遲 ${pressAt}ms（confirmMs=${C.confirmMs}）`);
  check('第三剎車燈判據可用', det.chUsable);
  check('狀態為 on', det.chState === 'on' && det.state === 'on');
  check('位置鎖在中央（x 應接近 0.5）',
    det.chPos && Math.abs(det.chPos.x - 0.5) < 0.06, `x=${det.chPos && det.chPos.x.toFixed(3)}`);

  const tOff = t;
  for (; t < tOff + 1500; t += 100) {
    const r = det.updateFromStats(DARK, t, 270);
    if (r.released && relAt === null) relAt = t - tOff;
  }
  check('鬆開剎車 → 偵測到「鬆開」', relAt !== null, `延遲 ${relAt}ms`);
  check('動態範圍遠超門檻', det.chPeak / Math.max(det.chFloor, 1) > C.rangeMin,
    `峰值 ${det.chPeak.toFixed(0)} / 谷值 ${det.chFloor.toFixed(0)}`);

  // 車太遠 → 這顆燈只剩幾個像素，且搜尋區會吃到背景 → 不使用這條判據
  const det2 = new BrakeLightDetector(CONFIG);
  for (let k = 0; k < 20; k++) det2.updateFromStats(LIT, k * 100, C.minBoxW - 20);
  check(`框寬 < ${C.minBoxW}px（車太遠）→ 不使用第三剎車燈`,
    !det2.chUsable && det2.chFar);
}

console.log('');
console.log('=== 中央的第三剎車燈不得汙染「車身參考區」 ===');
// 離線跑機在實車影片上抓到的設計矛盾：「有一對紅燈」的判定用
// 「燈區紅度 / 中央車身紅度」當對比，而**法規規定第三剎車燈就裝在車後中線**。
// 實測那台車：左燈 52、右燈 56、中央車身 67 → 對比 0.72 < 門檻 1.35
// → 判成「沒有一對紅燈」，而第三剎車燈的網格值是 193。
// 修法：參考區改取「燈帶以下」（第三剎車燈依法在煞車燈之上）。
{
  const d = new Uint8ClampedArray(W * H * 4);
  const put = (x, y, c) => {
    const i = (y * W + x) * 4;
    d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) put(x, y, GREY);
  // 左右尾燈（在燈帶內）
  for (let y = Math.round(H * 0.45); y < Math.round(H * 0.75); y++) {
    for (let x = Math.round(W * 0.04); x < Math.round(W * 0.25); x++) put(x, y, TAIL);
    for (let x = Math.round(W * 0.75); x < Math.round(W * 0.96); x++) put(x, y, TAIL);
  }
  // 中央上方的第三剎車燈：一條橫跨中線的紅色燈條
  for (let y = Math.round(H * 0.04); y < Math.round(H * 0.12); y++) {
    for (let x = Math.round(W * 0.38); x < Math.round(W * 0.62); x++) put(x, y, BRAKE);
  }
  const st = lampStats(d, W, H, B);
  const contrast = Math.min(st.left, st.right) / (st.body + 6);
  console.log(`     左燈 ${st.left.toFixed(0)}　右燈 ${st.right.toFixed(0)}`
    + `　中央車身 ${st.body.toFixed(0)}（參考區 y ${B.bodyYTop}~${B.bodyYBottom}）`
    + `　對比 ${contrast.toFixed(2)}`);
  check('車身參考區沒有吃到中央的第三剎車燈', st.body < 20, `body=${st.body.toFixed(0)}`);
  check('因此「有一對紅燈」判定成立', contrast >= B.minContrast);
}

console.log('');
console.log('=== 鎖錯位置要能自己恢復 ===');
// 實測踩過的坑：搜尋範圍原本設得太寬（x ±22%），右邊界碰到**右尾燈**，
// 於是在第三燈還沒亮的時候就把追蹤位置鎖在尾燈上（值 68），
// 之後真正的燈亮起（值 142）也讀不到 —— 因為它離追蹤位置太遠。
// 搜尋範圍收窄後這個情形不會再由尾燈造成，但背景紅光仍可能出現在任何位置，
// 所以「出現明顯更強的候選就重新鎖定」這條恢復路徑必須有效。
{
  const C = B.chmsl;
  const mk = (cells, bg) => {
    const vals = new Float32Array(C.cols * C.rows).fill(bg);
    for (const [rx, ry, v] of cells) vals[ry * C.cols + rx] = v;
    const h = new Int32Array(32); h[7] = 1200; h[1] = 2800;
    return {
      left: 60, right: 60, body: 20,
      histL: h, histR: h, nL: 4000, nR: 4000,
      chGrid: {
        vals, cols: C.cols, rows: C.rows,
        cellW: C.searchXFrac / C.cols, cellH: C.searchYFrac / C.rows,
        xOff: 0.5 - C.searchXFrac / 2,
      },
      overexposed: 0.02, luma: 62, n: 12000,
    };
  };
  const det = new BrakeLightDetector(CONFIG);
  let t = 0;
  // 先讓某個邊緣格出現 68（模擬背景紅光）→ 位置被鎖在那裡
  for (; t < 1000; t += 100) det.updateFromStats(mk([[11, 3, 68]], 11), t, 270);
  const wrongX = det.chPos && det.chPos.x;
  check('先鎖在錯誤位置', det.chPos !== null && Math.abs(wrongX - 0.5) > 0.08,
    `x=${wrongX && wrongX.toFixed(3)}`);

  // 真正的第三燈在中央亮起（142），錯誤位置只剩 20
  for (; t < 3000; t += 100) det.updateFromStats(mk([[4, 0, 142], [11, 3, 20]], 15), t, 270);
  check('出現更強候選 → 重新鎖定到中央',
    det.chPos && Math.abs(det.chPos.x - 0.5) < 0.06, `x=${det.chPos && det.chPos.x.toFixed(3)}`);
  check('重新鎖定後狀態跟上', det.chState === 'on', det.lastDetail.split('\n')[1]);
  // 重新鎖定當下燈已經亮著 → 範圍未解析 → 還不敢把它當剎車燈的邊緣事件。
  // 但燈一熄，谷值就掉到 0、範圍在同一個畫格解析，「鬆開」照樣抓得到。
  check('重新鎖定當下範圍還沒解析（誠實回報不可用）', !det.chUsable);
  let rel = false;
  for (; t < 5000; t += 100) {
    if (det.updateFromStats(mk([[4, 0, 0], [11, 3, 20]], 6), t, 270).released) rel = true;
  }
  check('燈熄之後仍抓到「鬆開」', rel && det.chState === 'off', det.lastDetail.split('\n')[1]);
}

console.log('');
console.log('=== 第三剎車燈可用時，它蓋過外側燈的判定 ===');
// 夜間外側燈全程亮著（尾燈），單看它只有 1.5 倍落差、而且分不出尾燈/剎車。
// 第三剎車燈可用時應該直接由它決定。
{
  const C = B.chmsl;
  const mk = (ch, chBg) => {
    const vals = new Float32Array(C.cols * C.rows).fill(chBg);
    for (const rx of [4, 5]) for (const ry of [0, 1]) vals[ry * C.cols + rx] = ch;
    const h = new Int32Array(32); h[15] = 1200; h[1] = 2800;   // 外側燈固定亮著
    return {
      left: 122, right: 122, body: 61,
      histL: h, histR: h, nL: 4000, nR: 4000,
      chGrid: {
        vals, cols: C.cols, rows: C.rows,
        cellW: C.searchXFrac / C.cols, cellH: C.searchYFrac / C.rows,
        xOff: 0.5 - C.searchXFrac / 2,
      },
      overexposed: 0.02, luma: 62, n: 12000,
    };
  };
  const det = new BrakeLightDetector(CONFIG);
  let t = 0;
  for (; t < 2000; t += 100) det.updateFromStats(mk(145, 23), t, 270);
  // 鎖定時燈就已經亮著（停在正在踩剎車的車後面）→ 峰值 = 谷值、範圍未解析，
  // 還分不出那是剎車燈還是中央的紅色貼紙 → 誠實地維持外側燈的 lit
  check('鎖定時已亮著 → 先回報 lit（範圍未解析）',
    det.state === 'lit' && !det.chUsable, det.lastDetail.split('\n')[1]);
  const outerBefore = det.levelL;
  let released = false;
  for (; t < 4000; t += 100) if (det.updateFromStats(mk(0, 6), t, 270).released) released = true;
  check('第三燈熄 → 即使外側燈完全沒變也判定鬆開', released && det.state === 'off',
    `外側燈位準全程 ${outerBefore.toFixed(0)} → ${det.levelL.toFixed(0)}（沒變）`);
  check('此時第三燈判據已可用（轉換本身就是證據）', det.chUsable);
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
console.log('=== 框漂移時，第三剎車燈不得被讀成熄滅 ===');
// 2026-09-10 夜間錄影量到的：第三剎車燈裝在車頂線上，而 YOLO 的框上緣
// 「就是」車頂線 —— 兩者重疊在同一條線上，一點餘裕都沒有。
//   t=5s（實線框，剛偵測）  燈在 bbox 內 y=0.036
//   t=23s（虛線框，KF 外推）燈在 bbox **上方 27px**（y=−0.16）
// 那 27 秒裡燈的紅度是 160~183 的常數（直接量錄影的像素，與 app 無關），
// 一次都沒有熄過，app 卻反覆報「鬆開剎車 / 踩下剎車」。
//
// 兩道修法在這裡一起驗證：
//   (a) 裁切往 bbox 上緣之外多留 topMarginFrac → 燈不會掉出搜尋範圍
//   (b) 追蹤位置存**影像座標** → 框怎麼漂，取樣點都還在同一顆燈上
{
  const C = B.chmsl;
  const BOX_W = 270, BOX_H = 250, BOX_X = 380;
  const LAMP_X = BOX_X + BOX_W / 2;          // 車後中線
  const LAMP_Y = 600 + 0.032 * BOX_H;        // 貼著車頂線（實測 y=0.036）

  /** 依「當下的 bbox 位置」組出裁切矩形 + 對應的網格（燈固定在影像座標） */
  const mkAt = (boxY, chVal, bg) => {
    const m = C.topMarginFrac * BOX_H;
    const cropY = Math.max(0, boxY - m);
    const cropH = BOX_H + (boxY - cropY);
    const geom = { yOff: (boxY - cropY) / cropH, yScale: BOX_H / cropH };
    const cellW = C.searchXFrac / C.cols;
    const bandH = geom.yOff + C.searchYFrac * geom.yScale;   // 搜尋帶佔裁切圖的比例
    const cellH = bandH / C.rows;
    const xOff = 0.5 - C.searchXFrac / 2;
    const vals = new Float32Array(C.cols * C.rows).fill(bg);
    // 燈落在哪一格，完全由「影像座標 → 裁切座標」決定
    const nx = (LAMP_X - BOX_X) / BOX_W, ny = (LAMP_Y - cropY) / cropH;
    const cx = Math.floor((nx - xOff) / cellW), cy = Math.floor(ny / cellH);
    let inBand = false;
    if (cx >= 0 && cx < C.cols && cy >= 0 && cy < C.rows) {
      vals[cy * C.cols + cx] = chVal;
      inBand = true;
    }
    const h = new Int32Array(32); h[15] = 1200; h[1] = 2800;  // 外側尾燈全程亮著
    return {
      st: {
        left: 122, right: 122, body: 61,
        histL: h, histR: h, nL: 4000, nR: 4000,
        chGrid: { vals, cols: C.cols, rows: C.rows, cellW, cellH, xOff },
        overexposed: 0.02, luma: 62, n: 12000,
      },
      crop: { x: BOX_X, y: cropY, w: BOX_W, h: cropH },
      inBand,
    };
  };

  check('框沒漂時燈在搜尋帶內', mkAt(600, 170, 20).inBand);
  check('框往下漂 0.16 個框高，燈仍在搜尋帶內（靠 topMarginFrac 的餘裕）',
    mkAt(600 + 0.16 * BOX_H, 170, 20).inBand,
    `topMarginFrac=${C.topMarginFrac}`);

  const det = new BrakeLightDetector(CONFIG);
  let t = 0;
  // 先亮一段再熄一次 —— 讓動態範圍解析，這條判據才會被採用
  for (; t < 1500; t += 100) { const a = mkAt(600, 170, 20); det.updateFromStats(a.st, t, BOX_W, a.crop); }
  for (; t < 3000; t += 100) { const a = mkAt(600, 0, 6); det.updateFromStats(a.st, t, BOX_W, a.crop); }
  for (; t < 5000; t += 100) { const a = mkAt(600, 170, 20); det.updateFromStats(a.st, t, BOX_W, a.crop); }
  check('燈亮著且範圍已解析 → on', det.chUsable && det.chState === 'on',
    String(det.lastDetail).split(String.fromCharCode(10))[1]);

  // ---- 燈一直亮著，只有框在漂 ----
  let released = false, minVal = Infinity;
  for (let k = 0; k < 30; k++, t += 100) {
    const boxY = 600 + (k / 29) * 0.16 * BOX_H;      // 慢慢往下漂到 0.16 個框高
    const a = mkAt(boxY, 170, 20);
    const r = det.updateFromStats(a.st, t, BOX_W, a.crop);
    if (r.released) released = true;
    minVal = Math.min(minVal, det.chVal);
  }
  check('框漂移 0.16 個框高時不會誤報「鬆開剎車」', !released && det.chState === 'on',
    `期間讀到的最低值 ${minVal.toFixed(0)}（燈全程 170）`);

  // ---- 框逐幀抖動（YOLO 的框每幀都在抖）----
  let released2 = false;
  for (let k = 0; k < 40; k++, t += 100) {
    const a = mkAt(600 + 0.16 * BOX_H + (k % 2 ? 8 : -8), 170, 20);
    const r = det.updateFromStats(a.st, t, BOX_W, a.crop);
    if (r.released) released2 = true;
  }
  check('框逐幀抖動 ±8px 時不會誤報「鬆開剎車」', !released2 && det.chState === 'on');

  // ---- 真的熄了還是要抓得到 ----
  let released3 = false;
  for (let k = 0; k < 20; k++, t += 100) {
    const a = mkAt(600 + 0.16 * BOX_H, 0, 6);
    if (det.updateFromStats(a.st, t, BOX_W, a.crop).released) released3 = true;
  }
  check('燈真的熄滅時仍然抓得到「鬆開」', released3 && det.chState === 'off');
}

console.log('');
if (fails) {
  console.log(`❌ ${fails} 項未通過`);
  process.exitCode = 1;
} else {
  console.log('✅ 全部通過');
}
