// BYTE 兩段式關聯的行為測試
//
// 為什麼需要它：2026-09-10 那支停等車陣的實測，前車的偵測信心**中位數
// 只有 0.19**，而 confThreshold = 0.30 —— 同一批畫格只改門檻：
//     conf ≥ 0.30 → 找到前車 22%
//     conf ≥ 0.05 → 69%
// 整條管線 78% 的時間看不到前車，下游的目標換手、光流重新錨定、
// 剎車燈 ROI 漂掉全部是這一件事的後果。
//
// ByteTrack（ECCV 2022）的作法是：低分框只跟**既有 track** 配（只用 IoU），
// 沒配上就丟掉。要驗證的是兩件事同時成立：
//   (a) 低分框真的把軌跡接起來了（不再碎裂）
//   (b) 低分的**背景**框不會憑空生出 track

import { Tracker } from '../../src/tracking/tracker.js';
import { CONFIG } from '../../src/config.js';

let fails = 0;
const check = (name, ok, note = '') => {
  if (!ok) fails++;
  console.log(`  ${ok ? '✅' : '✗ '} ${name}${note ? '  ' + note : ''}`);
};

const car = (x, y, w, h, score) => ({ x, y, w, h, score, classId: 2 });

console.log('=== 低分框把碎裂的軌跡接起來 ===');
// 一台停著的車，偵測信心在 0.30 附近跳動（實測中位數 0.19）：
// 只有第 0、1 幀是高分，之後全部落在低分層。
{
  const tk = new Tracker(CONFIG);
  let ts = 0;
  tk.update([car(200, 400, 280, 240, 0.55)], ts); ts += 125;
  tk.update([car(202, 401, 279, 241, 0.51)], ts); ts += 125;
  const id = tk.tracks[0].id;
  check('兩幀高分 → track 已確認', tk.tracks[0].confirmed);

  // 之後 16 幀（2 秒）全部是低分框
  for (let k = 0; k < 16; k++, ts += 125) {
    tk.update([car(200 + (k % 3), 400 + (k % 2), 280, 240, 0.18)], ts);
  }
  check('低分框延續了同一個 track（沒有換 id）',
    tk.tracks.length === 1 && tk.tracks[0].id === id,
    `id ${id} → ${tk.tracks.map((t) => t.id).join(',')}`);
  check('2 秒後仍然是「新鮮」的（maxCoastMs=1500ms）',
    ts - tk.tracks[0].lastSeenTs < 200,
    `距上次觀測 ${ts - tk.tracks[0].lastSeenTs}ms`);
  check('低分框有被記錄成回收', tk.stats.recoveredLow === 16,
    `recoveredLow=${tk.stats.recoveredLow}`);

  // 對照組：關掉第二段（把低分框直接丟掉）會怎樣
  const tk2 = new Tracker(CONFIG);
  let ts2 = 0;
  tk2.update([car(200, 400, 280, 240, 0.55)], ts2); ts2 += 125;
  tk2.update([car(202, 401, 279, 241, 0.51)], ts2); ts2 += 125;
  for (let k = 0; k < 16; k++, ts2 += 125) tk2.update([], ts2);   // 等同丟掉低分框
  check('對照組：丟掉低分框 → track 被淘汰', tk2.tracks.length === 0);
}

console.log('');
console.log('=== 低分的背景框不得憑空生出 track ===');
// 護欄、反光、招牌在夜間常常拿到 0.1~0.2 的分數。它們沒有對應的既有軌跡，
// 所以必須被丟掉 —— 這正是「不降低整體門檻」的理由。
{
  const tk = new Tracker(CONFIG);
  let ts = 0;
  for (let k = 0; k < 10; k++, ts += 125) {
    tk.update([
      car(20 + k * 3, 700, 60, 40, 0.14),      // 路邊的假框，每幀位置都在飄
      car(480, 300 + k * 5, 50, 50, 0.11),
    ], ts);
  }
  check('全是低分框 → 一個 track 都沒建立', tk.tracks.length === 0,
    `tracks=${tk.tracks.length} created=${tk.stats.created}`);
  check('低分框有被計數但沒被採用', tk.stats.detsLow === 20 && tk.stats.recoveredLow === 0,
    `detsLow=${tk.stats.detsLow} recoveredLow=${tk.stats.recoveredLow}`);
}

console.log('');
console.log('=== 低分框不能把「未確認」的 track 拱成前車候選 ===');
// 只被高分框看過一次的 track 還不算數（confirmHits=2）。
// 若低分框也能累積命中數，一個雜訊框 + 幾個低分框就會變成「確認的前車」。
{
  const tk = new Tracker(CONFIG);
  let ts = 0;
  tk.update([car(200, 400, 280, 240, 0.55)], ts); ts += 125;      // 1 次高分
  check('一次高分 → 尚未確認', !tk.tracks[0].confirmed);
  for (let k = 0; k < 8; k++, ts += 125) {
    tk.update([car(200, 400, 280, 240, 0.19)], ts);
  }
  check('之後全是低分框 → 仍然未確認（低分不累積 confirmHits）',
    tk.tracks.length === 1 && !tk.tracks[0].confirmed,
    `hits=${tk.tracks[0].hits}`);
  check('但軌跡有被維持住（沒被淘汰）', tk.tracks.length === 1);
  tk.update([car(200, 400, 280, 240, 0.61)], ts);
  check('再來一次高分 → 確認', tk.tracks[0].confirmed);
}

console.log('');
console.log('=== 第二段只用 IoU，門檻比第一段緊 ===');
// 第一段有「中心距」這條寬鬆的救援路徑（IoU 掉到 0 也能配）。
// 低分框不給這條路：它本來就比較可能是背景。
{
  const t = CONFIG.tracker;
  check(`iouGateLow (${t.iouGateLow}) 比 iouGate (${t.iouGate}) 緊`,
    t.iouGateLow > t.iouGate);

  const tk = new Tracker(CONFIG);
  let ts = 0;
  tk.update([car(200, 400, 200, 200, 0.55)], ts); ts += 125;
  tk.update([car(200, 400, 200, 200, 0.55)], ts); ts += 125;
  const id = tk.tracks[0].id;
  // 一個離很遠、IoU=0 但中心距在 1.2×對角線內的低分框
  tk.update([car(400, 480, 200, 200, 0.15)], ts);
  check('IoU 太低的低分框不會被第二段吃進來', tk.stats.recoveredLow === 0,
    `recoveredLow=${tk.stats.recoveredLow}`);
  check('也不會建立新 track', tk.tracks.length === 1 && tk.tracks[0].id === id);
}

console.log('');
console.log('=== 原始量測框有被記下來（bbox 尺度那條路要用）===');
{
  const tk = new Tracker(CONFIG);
  tk.update([car(200, 400, 280, 240, 0.55)], 0);
  tk.update([car(210, 405, 260, 220, 0.55)], 125);
  const tr = tk.tracks[0];
  check('lastBox 是原始偵測框，不是 KF 外推值',
    tr.lastBox.w === 260 && tr.lastBox.h === 220 && tr.lastBoxTs === 125,
    `lastBox ${tr.lastBox.w}x${tr.lastBox.h} @${tr.lastBoxTs}ms`);
}

console.log('');
if (fails) {
  console.log(`❌ ${fails} 項未通過`);
  process.exitCode = 1;
} else {
  console.log('✅ 全部通過');
}
