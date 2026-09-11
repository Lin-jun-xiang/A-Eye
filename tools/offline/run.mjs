// 用 stdin 的 rawvideo 幀驅動真正的 Pipeline，把每一步的內部狀態印出來。
//
// 用法（見 README.md）：
//   ffmpeg -v error -i in.mp4 -ss 148 -t 18 -vf "fps=10,crop=588:940:0:0" \
//     -pix_fmt rgba -f rawvideo pipe:1 | node run.mjs 588 940 10 148
import { installShims, loadCv, NodeDetector, DetrDetector, frameReader } from './harness.mjs';
import { createRequire } from 'module';
import { CONFIG } from '../../src/config.js';
import { Pipeline } from '../../src/core/pipeline.js';

const require_ = createRequire(import.meta.url);
const [W, H, fps, t0] = process.argv.slice(2, 6).map(Number);
if (!W || !H || !fps) {
  console.error('用法: node run.mjs <寬> <高> <fps> <起始秒> （rawvideo rgba 從 stdin 進來）');
  process.exit(2);
}
const FRAME = W * H * 4;
const CV = process.env.CV_JS || new URL('./opencv.js', import.meta.url).pathname.replace(/^\//, '');
// 預設與 config.js 的 modelCandidates[0] 一致，否則跑機量到的不是 app 的行為
const MODEL = process.env.MODEL || new URL('../../models/yolov8s.onnx', import.meta.url).pathname.replace(/^\//, '');

installShims();
process.stderr.write('載入 opencv.js...\n');
await loadCv(CV);
process.stderr.write('載入模型...\n');
const ort = require_('onnxruntime-node');
// MODEL=... 指定模型；FAMILY=detr 換成 DETR 那條解碼路徑
// （預設由 config.js 的 detector.family 決定，與 app 一致）
const FAMILY = process.env.FAMILY || CONFIG.detector.family;
if (FAMILY === 'detr') CONFIG.detector.family = 'detr';
const det = FAMILY === 'detr'
  ? new DetrDetector(CONFIG, process.env.MODEL
      || new URL('../../models/detr-resnet-50-fp16.onnx', import.meta.url).pathname.replace(/^\//, ''), ort)
  : new NodeDetector(CONFIG, MODEL, ort);
const info = await det.init();
process.stderr.write(`模型 ${info.model} @${info.provider} ${info.inputSize}px\n`);

const pipeline = new Pipeline({ cfg: CONFIG, gps: null, imu: null });
// 影片沒有 GPS/IMU：預設讓判定自己用背景尺度率判斷自車動不動，
// ASSUME_STILL=1 則強制假設靜止（適合只想看前車那條路徑的片段）。
pipeline.assumeStill = process.env.ASSUME_STILL === '1';
await pipeline.initCv(async () => {});

const next = frameReader(process.stdin, FRAME);
// DETECT_HZ=1.2 模擬實機的偵測率。這個開關的存在是一次教訓：
// 離線跑機預設 detectHz=8，而手機上 DETR@480 fp16 實測只有 1.2Hz ——
// 用 8Hz 驗證過的「bbox 尺度路徑會觸發」在實機上根本湊不滿 600ms 窗口
//（3 筆偵測 @1.2Hz = 2.5 秒），量測間隔直接掉進 SPRT 的不可達區。
// 凡是與量測率有關的結論，必須在 DETECT_HZ=1.2 下重跑一次才算數。
const detGap = 1000 / (Number(process.env.DETECT_HZ) || CONFIG.loop.detectHz);
// CFG='{"tracker":{"coastVelocityTauMs":1e9}}' —— 對 CONFIG 做深度合併覆寫。
// 用途：二分定位（「這個回歸是哪個參數造成的？」）不必改碼重跑。
if (process.env.CFG) {
  const merge = (dst, src) => {
    for (const k of Object.keys(src)) {
      if (src[k] && typeof src[k] === 'object' && !Array.isArray(src[k])) merge(dst[k] ??= {}, src[k]);
      else dst[k] = src[k];
    }
  };
  merge(CONFIG, JSON.parse(process.env.CFG));
  process.stderr.write(`CFG 覆寫: ${process.env.CFG}\n`);
}
let lastDet = -Infinity, f = 0;
const events = [];

for (;;) {
  const buf = await next();
  if (!buf) break;
  const img = { width: W, height: H, data: new Uint8ClampedArray(buf) };
  const now = (t0 + f / fps) * 1000;

  // 偵測依 detectHz 在影片時間上節流（與即時路徑相同的政策）
  if (now - lastDet >= detGap) {
    lastDet = now;
    pipeline.onDetections(await det.detect(img), now);
  }

  const { events: evs, hud } = pipeline.tick({ source: img, vw: W, vh: H, now });
  for (const e of evs) events.push({ t: now / 1000, kind: e.kind, text: e.text });

  if (f % Math.round(fps) === 0) {
    const d = hud.departure, fl = pipeline.lastFlow, st = hud.stats;
    console.log(`t=${(now / 1000).toFixed(1)}s  target=${hud.target ? '#' + hud.target.id : '--'}`
      + `${hud.target ? ` ${Math.round(hud.target.box.w)}x${Math.round(hud.target.box.h)}` : ''}`
      + `  新鮮${st.targetTicks ? (st.targetFresh / st.targetTicks * 100).toFixed(0) : '--'}%`
      + `  ego=${hud.ego}`
      + `  z=${d.z.toFixed(2)}/${d.zFire.toFixed(2)} LLR=${d.llr.toFixed(1)}/${d.sprtA.toFixed(1)}`
      + ` V=${d.V.toFixed(3)} n=${d.ticks} ${d.reason}`
      + `  flow=${fl ? (fl.ok ? `ok fg=${fl.fg.nIn}/${fl.fg.n} bg=${fl.bg.nIn}/${fl.bg.n}` : fl.reason) : '--'}`
      + `  bbox=${pipeline.bboxScale.lastReason}`
      + `  剎車燈=${hud.brakeState}${hud.brakeChmslUsable ? `(第三燈${hud.brakeChmslState})` : ''}`);
  }
  f++;
}

const st = pipeline.stats, tk = pipeline.tracker.stats;
console.log('\n===== 總結 =====');
console.log(`幀數 ${f}（${t0.toFixed(0)}~${(t0 + (f - 1) / fps).toFixed(1)}s @${fps}fps）`);
console.log(`偵測 ${st.detections} 次，高分框 ${tk.dets} 個 → 配對 ${tk.matched} 新建 ${tk.created} 淘汰 ${tk.dropped}`);
console.log(`低分框 ${tk.detsLow} 個 → BYTE 第二段救回 ${tk.recoveredLow}`);
console.log(`目標新鮮 ${st.targetTicks ? (st.targetFresh / st.targetTicks * 100).toFixed(0) : '--'}%`
  + `（${st.targetFresh}/${st.targetTicks}）　換手 ${st.targetChanges} 次（清空證據 ${st.evidenceResets} 次）`);
const fails = Object.entries(st.flowFail || {}).sort((a, b) => b[1] - a[1]);
console.log(`光流 ok ${st.flowOk}　失敗：${fails.map(([k, v]) => `${k}:${v}`).join(' ') || '無'}`);
console.log(`bbox 尺度量測（光流交白卷時的第二條路）${st.bboxOk} 筆`);
console.log(`事件 ${events.length} 個：`);
for (const e of events) console.log(`   ${e.t.toFixed(2)}s  ${e.text}`);
