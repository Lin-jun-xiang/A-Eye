// 傾印「畫面上那個目標框」逐 tick 的變化 + track 生死
import { installShims, loadCv, DetrDetector, frameReader } from './harness.mjs';
import { createRequire } from 'module';
import { CONFIG } from '../../src/config.js';
import { Pipeline } from '../../src/core/pipeline.js';
const require_ = createRequire(import.meta.url);
const [W,H,fps,t0] = process.argv.slice(2,6).map(Number);
installShims();
await loadCv(new URL('./opencv.js', import.meta.url).pathname.replace(/^\//,''));
const ort = require_('onnxruntime-node');
CONFIG.detector.family='detr';
const det=new DetrDetector(CONFIG,new URL('../../models/detr-resnet-50-fp16.onnx',import.meta.url).pathname.replace(/^\//,''),ort);
await det.init();
const pipeline=new Pipeline({cfg:CONFIG,gps:null,imu:null});
pipeline.assumeStill=true; await pipeline.initCv(async()=>{});
const next=frameReader(process.stdin,W*H*4);
const detGap=1000/(Number(process.env.DETECT_HZ)||CONFIG.loop.detectHz);
let lastDet=-Infinity,f=0,prevIds=new Set();
let lastLine='';
for(;;){
  const buf=await next(); if(!buf)break;
  const img={width:W,height:H,data:new Uint8ClampedArray(buf)};
  const now=(t0+f/fps)*1000;
  if(now-lastDet>=detGap){ lastDet=now; pipeline.onDetections(await det.detect(img),now); }
  const {hud}=pipeline.tick({source:img,vw:W,vh:H,now});
  const ids=new Set(pipeline.tracker.tracks.map(t=>t.id));
  for(const id of prevIds) if(!ids.has(id)) console.log(`   t=${(now/1000).toFixed(2)}  track#${id} 死亡`);
  prevIds=ids;
  const tgt=hud.target;
  const line = tgt ? `#${tgt.id} w=${tgt.box.w.toFixed(0)} h=${tgt.box.h.toFixed(0)} x=${tgt.box.x.toFixed(0)}` : '--';
  if(line!==lastLine){ console.log(`t=${(now/1000).toFixed(2)}  目標 ${line}`); lastLine=line; }
  f++;
}
