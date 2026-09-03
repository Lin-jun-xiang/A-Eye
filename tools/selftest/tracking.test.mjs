// 追蹤 + 前車選取的行為測試
// 重點驗證 v6 的兩個漏報/誤判機制是否真的修掉了：
//   (a) 前車起步時 bbox 縮小 → 滯後的 EMA 框與真實框 IoU 掉破門檻 → 追蹤斷掉
//   (b) 車流中 ID switch → 鎖到旁車道車
//
// 場景一律用「地面平面投影」產生，確保幾何自洽：
//   給定距離 Z 與橫向偏移 X（公尺），
//     影像寬  w_px = f·W_car / Z
//     底邊 y  = y_horizon + f·h_cam / Z
//     中心 x  = cx0 + f·X / Z

import { Tracker } from '../../src/tracking/tracker.js';
import { FrontCarSelector } from '../../src/logic/frontCar.js';
import { CONFIG } from '../../src/config.js';
import { iou } from '../../src/util/math.js';

const VW = 1280, VH = 720;
const HFOV_DEG = 60;
const F = (VW / 2) / Math.tan((HFOV_DEG / 2) * Math.PI / 180);   // ≈ 1108 px
const H_CAM = CONFIG.frontCar.cameraHeightM;
const Y_HOR = CONFIG.frontCar.horizonFallback * VH;
const W_CAR = 1.8, H_CAR = 1.5;

let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const noise = (a) => (rnd() - 0.5) * 2 * a;

/** 由 (距離 Z, 橫向 X) 產生幾何自洽的偵測框 */
function project(Z, X, jitter = 0, classId = 2, score = 0.85) {
  const w = F * W_CAR / Z;
  const h = F * H_CAR / Z;
  const bottomY = Y_HOR + F * H_CAM / Z;
  const cx = VW / 2 + F * X / Z;
  return {
    x: cx - w / 2 + noise(jitter),
    y: bottomY - h + noise(jitter),
    w: w + noise(jitter * 1.5),
    h: h + noise(jitter * 1.5),
    classId, score,
  };
}

console.log(`場景參數：f=${F.toFixed(0)}px、相機高 ${H_CAM}m、地平線 y=${Y_HOR.toFixed(0)}px`);
console.log('');

// ---------- (a) 起步中的劇烈縮小：KF 預測 vs v6 的 EMA ----------
console.log('=== 起步時 bbox 快速縮小：KF 預測框 vs v6 的 EMA 平滑框 ===');
{
  const tr = new Tracker(CONFIG);
  const ids = new Set();
  let minIouKf = 1, minIouEma = 1;
  let ema = null;
  const EMA_ALPHA = 0.18;              // v6 的實際值
  const HZ = 8, DT = 1000 / HZ;

  for (let i = 0; i < 48; i++) {
    const ts = i * DT;
    const t = i / HZ;
    const Z = 8 + t * t;               // 8m 起步、2 m/s^2
    const d = project(Z, 0, 3);

    // 本版：KF 預測到 ts 的框
    if (tr.tracks.length) minIouKf = Math.min(minIouKf, iou(tr.tracks[0].boxAt(ts), d));
    tr.update([d], ts);
    tr.tracks.forEach((x) => ids.add(x.id));

    // v6 baseline：EMA 平滑框 + 原始 IoU 比對
    if (!ema) ema = { ...d };
    else {
      minIouEma = Math.min(minIouEma, iou(ema, d));
      ema = {
        x: ema.x + EMA_ALPHA * (d.x - ema.x),
        y: ema.y + EMA_ALPHA * (d.y - ema.y),
        w: ema.w + EMA_ALPHA * (d.w - ema.w),
        h: ema.h + EMA_ALPHA * (d.h - ema.h),
      };
    }
  }
  const gate = 0.15;                   // v6 的 IOU_LOCK_MATCH
  console.log(`  最低 IoU — KF 預測框 ${minIouKf.toFixed(3)} ／ v6 的 EMA 框 ${minIouEma.toFixed(3)}（v6 門檻 ${gate}）`);
  console.log(`  v6 是否會斷追：${minIouEma < gate ? '會 ✗' : '不會'}　本版是否會斷追：${minIouKf < CONFIG.tracker.iouGate ? '會 ✗' : '不會 ✓'}`);
  console.log(`  出現過的 track id 數 = ${ids.size} → ${ids.size === 1 ? '✅ 全程同一個 track' : '❌ 重建了 ' + (ids.size - 1) + ' 次'}`);
}

// ---------- (b) 偵測中斷（YOLO 漏檢）----------
console.log('');
console.log('=== 偵測中斷 375ms（YOLO 連續漏檢 3 幀）===');
{
  const tr = new Tracker(CONFIG);
  const HZ = 8, DT = 1000 / HZ;
  const ids = new Set();
  let survived = true;
  for (let i = 0; i < 30; i++) {
    const ts = i * DT;
    const gap = i >= 10 && i < 13;
    tr.update(gap ? [] : [project(10, 0, 2)], ts);
    if (gap && tr.tracks.length === 0) survived = false;
    tr.tracks.forEach((x) => ids.add(x.id));
  }
  console.log(`  中斷期間 track 是否存活：${survived ? '✅ 是（KF 慣性滑行）' : '❌ 否'}`);
  console.log(`  出現過的 id 數 = ${ids.size} → ${ids.size === 1 ? '✅ 沒有重建' : '❌ 重建過'}`);
}

// ---------- (c) 旁車道車 ID switch（幾何自洽版）----------
console.log('');
console.log('=== 本車道車靜止 + 旁車道車（橫向 3.5m）起步 ===');
{
  const tr = new Tracker(CONFIG);
  const sel = new FrontCarSelector(CONFIG);
  const HZ = 8, DT = 1000 / HZ;
  const picked = [];
  let frontId = null;
  for (let i = 0; i < 48; i++) {
    const ts = i * DT;
    const t = i / HZ;
    const front = project(10, 0, 2);                 // 本車道，靜止 10m
    const side = project(9.5 + t * t, 3.5, 2);       // 旁車道，正在起步
    tr.update([front, side], ts);
    const veh = tr.confirmedOf(CONFIG.vehicleClasses, ts);
    const target = sel.select(veh, VW, VH, ts);
    if (target) {
      picked.push(target.id);
      // 記錄「幾何上真的是本車道車」的那個 track（橫向偏移最小）
      let best = null, bd = Infinity;
      for (const v of veh) {
        const b = v.boxAt(ts);
        const dx = Math.abs((b.x + b.w / 2) - VW / 2);
        if (dx < bd) { bd = dx; best = v; }
      }
      if (best && frontId === null) frontId = best.id;
    }
  }
  const uniq = [...new Set(picked)];
  const w1 = sel.corridorWeight(project(10, 0), VW, VH);
  const w2 = sel.corridorWeight(project(10, 3.5), VW, VH);
  console.log(`  走廊權重：本車道車(X=0m) ${w1.toFixed(3)}、旁車道車(X=3.5m) ${w2.toFixed(3)}`);
  console.log(`  被選為前車的 id：${uniq.join(', ')}（幾何上的本車道車 id = ${frontId}）`);
  console.log(`  → ${uniq.length === 1 && uniq[0] === frontId ? '✅ 全程正確鎖定本車道車' : '❌ 選錯或發生 ID switch'}`);
}

// ---------- (d) 走廊的透視特性 ----------
console.log('');
console.log('=== 走廊半寬隨距離的變化（應為地平線處 0、往下線性放寬）===');
{
  const sel = new FrontCarSelector(CONFIG);
  sel.aspect = VH / VW;
  for (const Z of [30, 20, 10, 6, 4]) {
    const yr = (Y_HOR + F * H_CAM / Z) / VH;
    const hwFrac = sel.halfWidthAt(yr);
    const hwMeters = hwFrac * VW * Z / F;
    console.log(`  Z=${String(Z).padStart(2)}m → 底邊 y=${yr.toFixed(3)}、半寬 ${(hwFrac * 100).toFixed(1)}% 畫面寬`
      + ` = ${hwMeters.toFixed(2)}m（設定值 ${CONFIG.frontCar.laneHalfWidthM}m）`);
  }
  console.log('  （換算回公尺後在各距離都應等於設定值 → 證明幾何推導正確）');
}

console.log('');
console.log('=== 幀率對關聯穩健性的影響（v6 的實際 tick 率約 1~2Hz）===');
console.log('  Hz  | KF最低IoU | EMA最低IoU | v6會斷追(門檻0.15) | 本版會斷追');
for (const HZ of [1, 2, 4, 8, 20]) {
  const tr = new Tracker(CONFIG);
  const DT = 1000 / HZ;
  let minKf = 1, minEma = 1, ema = null, ids = new Set();
  const EMA_ALPHA = 0.18;
  for (let i = 0; i < Math.ceil(6 * HZ); i++) {
    const ts = i * DT;
    const t = i / HZ;
    const Z = 8 + t * t;
    const d = project(Z, 0, 3);
    if (tr.tracks.length) minKf = Math.min(minKf, iou(tr.tracks[0].boxAt(ts), d));
    tr.update([d], ts);
    tr.tracks.forEach((x) => ids.add(x.id));
    if (!ema) ema = { ...d };
    else {
      minEma = Math.min(minEma, iou(ema, d));
      ema = { x: ema.x + EMA_ALPHA * (d.x - ema.x), y: ema.y + EMA_ALPHA * (d.y - ema.y),
              w: ema.w + EMA_ALPHA * (d.w - ema.w), h: ema.h + EMA_ALPHA * (d.h - ema.h) };
    }
  }
  console.log(`  ${String(HZ).padStart(3)} |   ${minKf.toFixed(3)}   |   ${minEma.toFixed(3)}    |`
    + `        ${minEma < 0.15 ? '會 ✗' : '不會 '}       |   ${minKf < CONFIG.tracker.iouGate ? '會 ✗' : '不會 ✓'}`
    + `  (id數 ${ids.size})`);
}
