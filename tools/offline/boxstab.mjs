// 量測：(1) DETR 同幀重複框  (2) 目標 KF 框（畫面上那個）的高度/寬度穩定度
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
const iou=(a,b)=>{const x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y),
  x2=Math.min(a.x+a.w,b.x+b.w),y2=Math.min(a.y+a.h,b.y+b.h);
  const iw=x2-x1,ih=y2-y1; if(iw<=0||ih<=0)return 0;
  return iw*ih/(a.w*a.h+b.w*b.h-iw*ih);};
const next=frameReader(process.stdin,W*H*4);
const detGap=1000/(Number(process.env.DETECT_HZ)||CONFIG.loop.detectHz);
let lastDet=-Infinity,f=0,detFrames=0,dupPairs=0;
const hs=[],ws=[];        // 目標 KF 框逐 tick 序列（只取靜止期 t<2.0s）
for(;;){
  const buf=await next(); if(!buf)break;
  const img={width:W,height:H,data:new Uint8ClampedArray(buf)};
  const now=(t0+f/fps)*1000;
  if(now-lastDet>=detGap){
    lastDet=now;
    const ds=await det.detect(img);
    detFrames++;
    for(let i=0;i<ds.length;i++)for(let j=i+1;j<ds.length;j++)
      if(ds[i].classId===ds[j].classId&&iou(ds[i],ds[j])>0.7){
        dupPairs++;
        if(dupPairs<=4) console.log(`  dup t=${(now/1000).toFixed(1)}s IoU=${iou(ds[i],ds[j]).toFixed(2)} `
          +`${ds[i].w.toFixed(0)}x${ds[i].h.toFixed(0)}(${ds[i].score.toFixed(2)}) vs ${ds[j].w.toFixed(0)}x${ds[j].h.toFixed(0)}(${ds[j].score.toFixed(2)})`);
      }
    pipeline.onDetections(ds,now);
  }
  const {hud}=pipeline.tick({source:img,vw:W,vh:H,now});
  if(hud.target && now/1000 < 2.0){ hs.push(hud.target.box.h); ws.push(hud.target.box.w); }
  f++;
}
const cv_=a=>{const m=a.reduce((s,x)=>s+x,0)/a.length;
  return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/a.length)/m*100;};
console.log(`偵測幀 ${detFrames}  同類重複框(IoU>0.7) ${dupPairs} 對`);
console.log(`目標 KF 框（靜止期 <2s，n=${hs.length}）  寬 ${Math.min(...ws).toFixed(0)}~${Math.max(...ws).toFixed(0)} 變異 ${cv_(ws).toFixed(1)}%`
  +`   高 ${Math.min(...hs).toFixed(0)}~${Math.max(...hs).toFixed(0)} 變異 ${cv_(hs).toFixed(1)}%`);
