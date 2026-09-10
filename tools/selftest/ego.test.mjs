// 自車運動判定的行為測試
//
// 這裡最重要的情境是「GPS 靜止漂移」：2026-09-08 路測實測，停在路口不動時
// GPS 的 coords.speed 會跳到 2~5 km/h。原本的遲滯有一個致命的不對稱 ——
// 一個尖峰超過上緣（1.5 m/s = 5.4 km/h）就進入 MOVING，而退出需要掉到
// 0.5 m/s（1.8 km/h）以下，於是漂移把狀態鎖在 MOVING，整個紅燈都靜默。
//
// 而「靜默」是無聲的失效：畫面上只會顯示「行駛中 — 靜默中」，
// 使用者不會知道自己其實已經停下來了。

import { EgoMotionEstimator, EgoState } from '../../src/motion/egoMotion.js';
import { CONFIG } from '../../src/config.js';

let fails = 0;
const check = (label, cond, extra = '') => {
  console.log(`  ${cond ? '✅' : '✗ '} ${label}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

/** 假的 GPS：照給定的速度序列回放 */
class FakeGps {
  constructor(speeds, accuracy = 10) { this.speeds = speeds; this.i = 0; this.accuracy = accuracy; }
  currentSpeed() { return this.speeds[Math.min(this.i, this.speeds.length - 1)]; }
}

/** 依 config 的門檻，把舊版（無 dwell、無否決）的行為重現出來當對照 */
function legacy(speeds) {
  const e = CONFIG.ego;
  let moving = false;
  let stuckFrom = -1;
  speeds.forEach((sp, i) => {
    moving = moving ? sp > e.gpsStillSpeed : sp > e.gpsMoveSpeed;
    if (moving && stuckFrom < 0) stuckFrom = i;
  });
  return { endedMoving: moving, stuckFrom };
}

// 實測的漂移樣態：多數時間 2~4 km/h，偶爾一個 6.5 km/h 的尖峰
const CREEP = [];
for (let i = 0; i < 60; i++) {
  const kmh = (i === 12 || i === 41) ? 6.5 : 2 + (i % 7) * 0.45;   // 2.0~4.7 km/h
  CREEP.push(kmh / 3.6);
}

console.log('=== 停在路口，GPS 漂移 2~5 km/h（含兩個 6.5 km/h 尖峰）===');
{
  const old = legacy(CREEP);
  console.log(`  舊行為：第 ${old.stuckFrom} 筆就進入 MOVING，`
    + `結束時仍是 ${old.endedMoving ? 'MOVING（整個紅燈靜默）' : 'STILL'}`);
  check('對照組確認舊行為會卡在 MOVING', old.endedMoving);

  const gps = new FakeGps(CREEP);
  const ego = new EgoMotionEstimator(CONFIG, gps, null);
  let sawMoving = false;
  for (let i = 0; i < CREEP.length; i++) {
    gps.i = i;
    const st = ego.update(i * 200);        // 5Hz
    if (st === EgoState.MOVING) sawMoving = true;
  }
  check('新行為：全程未被判為行駛', !sawMoving, ego.debugLine());
  check('可以發出警示', ego.canAlert);
}

console.log('');
console.log('=== 真的開始行駛（持續 8 km/h 以上）仍要判為 MOVING ===');
{
  const speeds = new Array(40).fill(8 / 3.6);
  const gps = new FakeGps(speeds);
  const ego = new EgoMotionEstimator(CONFIG, gps, null);
  let firstMovingMs = null;
  for (let i = 0; i < speeds.length; i++) {
    gps.i = i;
    if (ego.update(i * 200) === EgoState.MOVING && firstMovingMs === null) firstMovingMs = i * 200;
  }
  check('會判為行駛', ego.state === EgoState.MOVING);
  check(`延遲約等於 gpsMoveDwellMs（${CONFIG.ego.gpsMoveDwellMs}ms）`,
    firstMovingMs !== null && firstMovingMs >= CONFIG.ego.gpsMoveDwellMs
      && firstMovingMs <= CONFIG.ego.gpsMoveDwellMs + 400, `${firstMovingMs}ms`);
  check('行駛中不得發出警示', !ego.canAlert);
}

console.log('');
console.log('=== 定位精度太差時，速度不可信 ===');
{
  const speeds = new Array(40).fill(9 / 3.6);
  const gps = new FakeGps(speeds, 80);          // accuracy 80m
  const ego = new EgoMotionEstimator(CONFIG, gps, null);
  for (let i = 0; i < speeds.length; i++) { gps.i = i; ego.update(i * 200); }
  check(`accuracy=80m > ${CONFIG.ego.gpsMaxAccuracyM}m → 不採用 GPS`,
    ego.source !== 'gps', `source=${ego.source} state=${ego.state}`);
}

console.log('');
console.log('=== 視覺否決：GPS 說在動，但背景尺度完全沒變 ===');
{
  const e = CONFIG.ego;
  // 速度剛好在漂移天花板之下、且持續超過上緣 → 沒有視覺證據時會判 MOVING
  const speeds = new Array(40).fill(2.5);       // 9 km/h < gpsCreepCeiling 2.8
  const gps = new FakeGps(speeds);

  const a = new EgoMotionEstimator(CONFIG, gps, null);
  for (let i = 0; i < 20; i++) { gps.i = i; a.update(i * 200); }
  check('沒有視覺證據 → 相信 GPS，判為 MOVING', a.state === EgoState.MOVING);

  const b = new EgoMotionEstimator(CONFIG, gps, null);
  for (let i = 0; i < 20; i++) { gps.i = i; b.update(i * 200, { bgExpZ: 0.4 }); }
  check('背景尺度 |z|=0.4 → 視覺否決，判為 STILL', b.state === EgoState.STILL, b.debugLine());

  const c = new EgoMotionEstimator(CONFIG, gps, null);
  for (let i = 0; i < 20; i++) { gps.i = i; c.update(i * 200, { bgExpZ: 12 }); }
  check('背景確實在逼近（z=12）→ 不否決，維持 MOVING', c.state === EgoState.MOVING);

  // 高速時不該讓視覺否決 —— 真實行駛的視覺證據若失效（背景太遠、夜間無紋理），
  // 否決會變成「行駛中照報警」，那是比漏報更危險的失效方向
  const fast = new FakeGps(new Array(40).fill(20));    // 72 km/h
  const d = new EgoMotionEstimator(CONFIG, fast, null);
  for (let i = 0; i < 20; i++) { fast.i = i; d.update(i * 200, { bgExpZ: 0.1 }); }
  check(`速度 > gpsCreepCeiling(${e.gpsCreepCeiling}m/s) → 不允許視覺否決`,
    d.state === EgoState.MOVING);
}

console.log('');
console.log(fails ? `❌ ${fails} 項未通過` : '✅ 全部通過');
if (fails) process.exitCode = 1;

console.log('');
console.log('=== 視覺判定自車運動：顯著 ≠ 夠大 ===');
// 2026-09-10 實測面板：`bgExpZ=8.0`、門檻 4.0 → ego=moving(visual-scale)
//   → canAlert=false → departure.update() 每 tick 走 reset()
//   → `V=0.000±100.000 n=0 ego-moving`，整個起步判定無聲關閉。
//
// 但 8.0 這個 z 換算回物理量是背景 1/TTC ≈ 0.01（TTC 100 秒）——
// 「10 公尺外的東西以 0.1 m/s 逼近」就是站著不動。
// σ_s 是近百個背景點的標準誤（~3e-4），所以靜止也能量出很大的 z。
//
// 反向誘因才是真正致命的地方：光流做得越準 → σ 越小 → 同一個物理靜止
// 產生越大的 z → 越容易被誤判成行駛中。
{
  const cfg = CONFIG;
  const mk = () => new EgoMotionEstimator(cfg, null, null);

  // (a) 實測那一組：z 很顯著，但物理上等於不動
  {
    const ego = mk();
    ego.update(1000, { resid: null, bgExpZ: 8.0, bgInvTtc: 0.01 });
    check('z=8.0 但背景 TTC 100 秒 → 判定靜止（警示不該被靜音）',
      ego.state === EgoState.STILL, `state=${ego.state} source=${ego.source}`);
    check('→ canAlert 為真', ego.canAlert);
  }

  // (b) 真的在動：兩個條件都成立
  {
    const ego = mk();
    ego.update(1000, { resid: null, bgExpZ: 8.0, bgInvTtc: 0.30 });
    check('z=8.0 且背景 TTC 3.3 秒 → 判定行駛中', ego.state === EgoState.MOVING,
      `state=${ego.state}`);
    check('→ canAlert 為假（行駛中不該報前車起步）', !ego.canAlert);
  }

  // (c) 物理量大但完全不顯著（雜訊）→ 不該說在動
  {
    const ego = mk();
    ego.update(1000, { resid: null, bgExpZ: 1.2, bgInvTtc: 0.40 });
    check('物理量大但 z 只有 1.2（雜訊）→ 仍判定靜止',
      ego.state === EgoState.STILL, `state=${ego.state}`);
  }

  // (d) 光流變準（σ 減半 → z 加倍）不得改變物理結論
  {
    const a = mk(), b = mk();
    a.update(1000, { resid: null, bgExpZ: 6.0, bgInvTtc: 0.02 });
    b.update(1000, { resid: null, bgExpZ: 60.0, bgInvTtc: 0.02 });   // 光流準 10 倍
    check('同一個物理靜止，z 從 6 變成 60，結論不變',
      a.state === b.state && a.state === EgoState.STILL,
      `${a.state} / ${b.state}`);
  }

  // (e) 沒有 bgInvTtc（舊介面）時退回只看 z —— 不能因為缺欄位就崩掉
  {
    const ego = mk();
    ego.update(1000, { resid: null, bgExpZ: 8.0 });
    check('舊介面（只有 bgExpZ）仍可運作', ego.state === EgoState.MOVING,
      `state=${ego.state}`);
  }
}

console.log('');
console.log('=== 視覺判定要有 dwell：單一雜訊 tick 不得翻面 ===');
// 實測 18 秒內 ego 在 still/moving 之間跳了 17 次，而每一次 moving
// 都讓下游的起步證據歸零（t=48s n=7 → t=49s n=0）。
{
  const ego = new EgoMotionEstimator(CONFIG, null, null);
  const still = { resid: null, bgExpZ: 1.0, bgInvTtc: 0.005 };
  const move = { resid: null, bgExpZ: 9.0, bgInvTtc: 0.30 };
  let t = 0;
  for (; t < 2000; t += 100) ego.update(t, still);
  check('先穩定在靜止', ego.state === EgoState.STILL);

  ego.update(t, move); t += 100;
  check('單一「在動」的 tick → 還不算行駛中', ego.state === EgoState.STILL,
    `state=${ego.state}`);
  ego.update(t, still); t += 100;
  check('雜訊過去後仍是靜止', ego.state === EgoState.STILL);

  const t0 = t;
  let switched = null;
  for (; t < t0 + 2000; t += 100) {
    ego.update(t, move);
    if (ego.state === EgoState.MOVING && switched === null) switched = t - t0;
  }
  check('持續「在動」→ 仍然會轉成行駛中', ego.state === EgoState.MOVING,
    `延遲 ${switched}ms（門檻 ${CONFIG.ego.visualMoveDwellMs}ms）`);
  check('轉換延遲符合 dwell 設定',
    switched !== null && switched >= CONFIG.ego.visualMoveDwellMs - 1
      && switched <= CONFIG.ego.visualMoveDwellMs + 200, `${switched}ms`);

  ego.update(t, still);
  check('退出行駛中不設 dwell（不對稱是刻意的）', ego.state === EgoState.STILL);
}
