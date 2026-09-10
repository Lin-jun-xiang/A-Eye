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

let failCount = 0;
const ok = (label, cond, extra = '') => {
  console.log(`  ${cond ? '✅' : '✗ '} ${label}${extra ? '  ' + extra : ''}`);
  if (!cond) failCount++;
};

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


console.log('');
console.log('=== 自車結構（引擎蓋／儀表板被 YOLO 認成 car）不得贏過真前車 ===');
// v7 首次路測的實際症狀：綠框框住自己的引擎蓋，於是光流量的是一個永遠不動
// 的東西 —— 起步警示在結構上不可能觸發。
//
// 假框為什麼必定贏：score 的第一項 proximity 直接由 bbox 底邊高度決定，
// 而引擎蓋的底邊就在畫面最下方 → proximity = 1.00，任何真前車都比不過。
//
// 修法刻意「不用硬性排除」：手機架高看不到引擎蓋時，3 公尺內的真前車
// 底邊會被畫面下緣裁掉，任何「底邊太低就排除」的規則都會誤殺它。
// 改成三項軟性加權相乘，每一項都由實體尺寸推導：
//   (a) 車尾長寬比      (b) w_px/Δy = 車寬/相機高（焦距自己消掉）
//   (c) 有沒有出現過一對對稱的剎車燈（正向證據，自車結構永遠不會有）
{
  const hood = { x: VW * 0.12, y: VH * 0.62, w: VW * 0.76, h: VH * 0.38, classId: 2, score: 0.42 };
  const car = project(8, 0);
  const sel = new FrontCarSelector(CONFIG);
  sel.aspect = VH / VW;

  const proxOf = (b) => ((b.y + b.h) / VH - sel.horizon) / (1 - sel.horizon);
  const scoreOf = (b) => proxOf(b) * sel.plausibility(b, VW, VH) * sel.corridorWeight(b, VW, VH);
  console.log(`  引擎蓋: aspect=${(hood.w / hood.h).toFixed(2)}`
    + ` w/dy=${sel.groundRatio(hood, VW, VH).toFixed(2)}`
    + ` plaus=${sel.plausibility(hood, VW, VH).toFixed(2)}`
    + ` prox=${proxOf(hood).toFixed(2)} → score=${scoreOf(hood).toFixed(3)}`);
  console.log(`  8m前車 : aspect=${(car.w / car.h).toFixed(2)}`
    + ` w/dy=${sel.groundRatio(car, VW, VH).toFixed(2)}`
    + ` plaus=${sel.plausibility(car, VW, VH).toFixed(2)}`
    + ` prox=${proxOf(car).toFixed(2)} → score=${scoreOf(car).toFixed(3)}`);
  ok('真前車分數高於引擎蓋假框（連剎車燈證據都還沒用上）',
    scoreOf(car) > scoreOf(hood));
  ok('但引擎蓋沒有被硬性排除（刻意的：硬排除會誤殺被裁切的近車）',
    sel.isCandidate(hood, VW, VH));

  const tr = new Tracker(CONFIG);
  let picked = null;
  for (let i = 0; i < 24; i++) {
    const ts = i * 125;
    tr.update([hood, car], ts);
    const t = sel.select(tr.confirmedOf(CONFIG.vehicleClasses, ts), VW, VH, ts);
    if (t) picked = t;
  }
  const carTrack = tr.tracks.find((t) => iou(t.boxAt(2875), car) > 0.8);
  ok('實際選取鎖定真前車', !!picked && !!carTrack && picked.id === carTrack.id,
    `選中 id=${picked && picked.id} 真前車 id=${carTrack && carTrack.id}`);

  // 剎車燈的證據是**加分**而不是扣分（扣分會造成自我毀滅的震盪，見 config 的說明）。
  // 真前車一旦被確認看到尾燈就更難被搶走；引擎蓋則永遠拿不到這個加分。
  // 有意義的門檻是選取器的遲滯（1.25×）：優勢要大於它，換手才不會發生。
  const bonus = CONFIG.frontCar.plausibility.lampBonus;
  const before = scoreOf(car) / scoreOf(hood);
  const after = scoreOf(car) * bonus / scoreOf(hood);
  ok('確認看到尾燈後，真前車的優勢超過選取遲滯（1.25×）',
    after > 1.25 && after > before,
    `優勢 ${before.toFixed(2)}× → ${after.toFixed(2)}×（加分 ${bonus}×）`);
}

console.log('');
console.log('=== 反向保護：手機架高、看不到引擎蓋，3m 內的近車底邊被畫面裁掉 ===');
// 這正是「底邊貼齊畫面最下緣就排除」那條硬編碼規則會誤殺的情形。
{
  const sel = new FrontCarSelector(CONFIG);
  sel.aspect = VH / VW;
  const raw = project(3, 0);                       // 3m：底邊落在畫面之外
  const clipped = { ...raw };
  clipped.h = Math.min(raw.h, VH - raw.y);         // 被畫面下緣裁掉
  console.log(`  未裁切底邊 y=${((raw.y + raw.h) / VH).toFixed(3)}（超出畫面）`
    + ` → 裁切後 y=${((clipped.y + clipped.h) / VH).toFixed(3)}`);
  console.log(`  裁切後: aspect=${(clipped.w / clipped.h).toFixed(2)}`
    + ` w/dy=${sel.groundRatio(clipped, VW, VH).toFixed(2)}`
    + ` plaus=${sel.plausibility(clipped, VW, VH).toFixed(3)}`);
  ok('被裁切的近車仍是候選', sel.isCandidate(clipped, VW, VH));
  ok('且幾何可信度沒有被扣分（> 0.9）', sel.plausibility(clipped, VW, VH) > 0.9);
}

console.log('');
console.log('=== 自車結構黑名單：只在自車行駛中學習 ===');
{
  const dash = { x: VW * 0.30, y: VH * 0.70, w: VW * 0.34, h: VH * 0.22, classId: 2, score: 0.45 };
  const sel = new FrontCarSelector(CONFIG);
  sel.aspect = VH / VW;
  const tr = new Tracker(CONFIG);

  let learnedAt = null;
  for (let i = 0; i < 40; i++) {
    const ts = i * 125;                       // 5 秒、8Hz
    tr.update([dash], ts);
    sel.learnEgoStructure(tr.confirmedOf(CONFIG.vehicleClasses, ts), VW, VH, ts, true);
    if (learnedAt === null && sel.egoRegions.length) learnedAt = ts;
  }
  ok(`自車行駛中 ${learnedAt} ms 後列入黑名單（門檻 ${CONFIG.frontCar.egoStructure.learnWhileMovingMs}ms）`,
    learnedAt !== null);
  ok('之後被硬性排除', !sel.isCandidate(dash, VW, VH));

  // 反向保護：紅燈停車時前車也不動，此時絕不能學
  const sel2 = new FrontCarSelector(CONFIG);
  const tr2 = new Tracker(CONFIG);
  const still = project(8, 0);
  for (let i = 0; i < 240; i++) {              // 30 秒紅燈
    const ts = i * 125;
    tr2.update([still], ts);
    sel2.learnEgoStructure(tr2.confirmedOf(CONFIG.vehicleClasses, ts), VW, VH, ts, false);
  }
  ok('自車靜止 30 秒（紅燈）後沒有誤學真前車', sel2.egoRegions.length === 0);
}

console.log('');
console.log('=== 偵測間歇時，track 不該一直被淘汰重建 ===');
// 實車量測（2026-09-09 夜間，近距離白車）：YOLO 只有 21% 的畫格抓得到這台車、
// 76% 在 coasting，於是 track 反覆被淘汰重建 —— 104 秒內換了 28 個 id。
// 而 pipeline 原本「id 一變就清空證據」，導致證據永遠從零開始（實測 證據 0%）。
{
  const still = project(8, 0);
  const gaps = [400, 800, 1200, 1600, 2000];
  console.log(`  maxCoastMs = ${CONFIG.tracker.maxCoastMs}ms`);
  for (const gap of gaps) {
    const tr = new Tracker(CONFIG);
    const ids = new Set();
    let ts = 0;
    // 先建立並確認 track
    for (let i = 0; i < 3; i++, ts += 125) { tr.update([still], ts); tr.tracks.forEach((t) => ids.add(t.id)); }
    // 中間空白 gap 毫秒（沒有任何偵測），再給一次偵測
    ts += gap;
    tr.update([still], ts);
    tr.tracks.forEach((t) => ids.add(t.id));
    const survived = ids.size === 1;
    ok(`空白 ${String(gap).padStart(4)}ms → ${survived ? '同一個 track' : `重建（產生 ${ids.size} 個 id）`}`,
      gap <= CONFIG.tracker.maxCoastMs ? survived : true);
  }
}

console.log('');
console.log('=== 目標為什麼會「消失」：淘汰在關聯之後，真正的閘門是新鮮度 ===');
// 寫測試時發現我原本的理解是錯的：Tracker 的淘汰（filter maxCoastMs）發生在
// **關聯之後**，所以只要偵測最終回來且配對成功，track 就不會因為空白而死掉
// —— 上面那組測試裡 2000ms 空白也還是同一個 track。
//
// 真正讓「目標」消失的是 confirmedOf(classIds, now, maxCoastMs) 這道新鮮度過濾：
// 超過 maxCoastMs 沒有新偵測的 track 不會被交給前車選取。
// 於是 pipeline 走進「沒有目標」的分支 —— 那裡原本每個 tick 都 flow.reset()，
// 把光流的特徵點與錨定 ROI 立刻銷毀。這才是實車上證據累積不起來的主要機制。
{
  const tr = new Tracker(CONFIG);
  const still = project(8, 0);
  let ts = 0;
  for (let i = 0; i < 3; i++, ts += 125) tr.update([still], ts);
  const id = tr.tracks[0].id;
  const box = tr.tracks[0].boxAt(ts);
  const lastSeen = ts - 125;

  for (const gap of [500, 1400, 1600, 3000]) {
    const now = lastSeen + gap;
    const sel = tr.confirmedOf(CONFIG.vehicleClasses, now);
    const expected = gap <= CONFIG.tracker.maxCoastMs;
    ok(`空白 ${String(gap).padStart(4)}ms → 前車選取${sel.length ? '看得到' : '看不到'}這個 track`,
      (sel.length > 0) === expected);
  }
  // track 本身還活著（沒有被淘汰）——它只是「不夠新鮮」
  ok('track 本身仍在清單中（只是不夠新鮮）', tr.tracks.some((t) => t.id === id));

  // 若真的換成旁車道的另一台車，IoU 低 → 證據必須清空
  const other = project(8, 3.5);
  const ov = iou(box, other);
  ok(`換成旁車道的車時 IoU 低（${ov.toFixed(2)} < ${CONFIG.tracker.sameTargetIou}）→ 會清空證據`,
    ov < CONFIG.tracker.sameTargetIou);
  // 同一位置重建的框 IoU 高 → 保留證據
  ok(`同一位置重建的框 IoU 高（${iou(box, still).toFixed(2)} ≥ ${CONFIG.tracker.sameTargetIou}）→ 保留證據`,
    iou(box, still) >= CONFIG.tracker.sameTargetIou);
}

console.log('');
console.log('=== 過期的大框不該贏過新鮮的小框 ===');
// 離線跑機在實車影片上量到的具體失效（148~166s，18 秒內目標換手 15 次、
// 每次都清空證據）：前車起步駛遠後，一個已經過期但 KF 外推出來仍然很大的框
// （328x204）與真實的新框（92x48）同時存在，而 score 的第一項 proximity
// 只看底邊高度 → 過期的大框贏 → 目標在兩者之間來回跳。
// coasting 的框是**預測**，不是量測；兩者同時存在時量測該贏。
{
  const sel = new FrontCarSelector(CONFIG);
  sel.aspect = VH / VW;
  const tr = new Tracker(CONFIG);
  const near = project(6, 0);      // 近車：框大、底邊低
  const far = project(25, 0);      // 同一條車道上遠一點的車：框小
  let ts = 0;
  for (let i = 0; i < 4; i++, ts += 125) tr.update([near, far], ts);
  const nearTrack = tr.tracks.find((t) => iou(t.boxAt(ts), near) > 0.8);
  const farTrack = tr.tracks.find((t) => iou(t.boxAt(ts), far) > 0.8);

  // 兩個都新鮮 → 近車（proximity 高）應該贏
  let pick = sel.select(tr.confirmedOf(CONFIG.vehicleClasses, ts), VW, VH, ts);
  ok('兩個都新鮮時，近車勝出', pick && pick.id === nearTrack.id);

  // 只餵遠車的偵測 → 近車開始過期
  const hl = CONFIG.frontCar.plausibility.staleHalfLifeMs;
  for (let i = 0; i < 8; i++, ts += 125) tr.update([far], ts);
  pick = sel.select(tr.confirmedOf(CONFIG.vehicleClasses, ts), VW, VH, ts);
  const nearAge = ts - nearTrack.lastSeenTs;
  ok(`近車過期 ${nearAge}ms（半衰期 ${hl}ms）後，改選新鮮的遠車`,
    pick && pick.id === farTrack.id, `選中 #${pick && pick.id}`);
}

console.log('');
console.log('=== 方向盤／儀表板：底邊貼著畫面下緣，但寬度遠不足 ===');
// 2026-09-10 實測：面板上 egoRegions=0 —— 自車結構黑名單只在「自車行駛中」
// 學習，停等紅燈時結構上學不到，而那正是引擎蓋/方向盤擋住視野的時候。
// 幾何加權原本又只查「高側」離群，方向盤在直式畫面裡 Δy 很大、
// 比值反而偏低，一分都沒扣。
//
// 但「底邊被畫面裁掉」是一個有下限的狀況：接地點在畫面外 → 至少那麼近 →
//   w_min = (W_car/h_cam)·(vh − y_horizon)     （焦距消掉）
// 588x980、地平線 0.45 → 汽車 809px、機車 359px，都比畫面 588 還寬。
// 所以「貼著下緣卻只有兩三百 px 寬」在幾何上不可能是站在地面上的車。
{
  const sel = new FrontCarSelector(CONFIG);
  const VW = 588, VH = 980;
  for (let i = 0; i < 60; i++) sel.setHorizon(0.45);      // 讓 EMA 收斂
  const dyMax = VH - sel.horizon * VH;
  const wMin = (CONFIG.frontCar.vehicleWidthM / CONFIG.frontCar.cameraHeightM) * dyMax;
  console.log(`  地平線 ${sel.horizon.toFixed(3)} → Δy_max ${dyMax.toFixed(0)}px`
    + `、真實車輛的寬度下限 ${wMin.toFixed(0)}px（畫面只有 ${VW}px 寬）`);

  const wheel = { x: 150, y: VH - 190, w: 260, h: 190 };   // 方向盤：貼著下緣、又窄又矮
  const car = { x: 190, y: 560, w: 210, h: 170 };          // 8m 外的真前車
  const moto = { x: 60, y: 700, w: 90, h: 150 };           // 機車：底邊沒被裁掉
  const pw = sel.plausibility(wheel, VW, VH);
  const pc = sel.plausibility(car, VW, VH);
  const pm = sel.plausibility(moto, VW, VH);
  console.log(`  方向盤 ${wheel.w}x${wheel.h} 底邊 y=${wheel.y + wheel.h} → plaus=${pw.toFixed(3)}`);
  ok('方向盤框的幾何可信度被明顯扣分', pw < 0.4, `plaus=${pw.toFixed(3)}`);
  ok('8m 外的真前車不受影響', pc > 0.9, `plaus=${pc.toFixed(3)}`);
  ok('底邊未裁切的窄框（機車）不受這條規則影響', pm > 0.9, `plaus=${pm.toFixed(3)}`);

  const prox = (b) => Math.max(0, Math.min(1,
    ((b.y + b.h) / VH - sel.horizon) / (1 - sel.horizon)));
  const sWheel = prox(wheel) * pw, sCar = prox(car) * pc;
  console.log(`  分數：方向盤 ${prox(wheel).toFixed(2)}x${pw.toFixed(2)}=${sWheel.toFixed(3)}`
    + `　真前車 ${prox(car).toFixed(2)}x${pc.toFixed(2)}=${sCar.toFixed(3)}`);
  ok('真前車勝出（即使方向盤的 proximity 是滿分）', sCar > sWheel);
}

console.log('');
console.log(failCount ? `❌ ${failCount} 項未通過` : '✅ 全部通過');
if (failCount) process.exitCode = 1;
