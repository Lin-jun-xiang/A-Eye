// =============================================
// 感知/判定管線（不接觸任何 DOM）
// =============================================
// 這是整個重構的關鍵：v6 把偵測邏輯、canvas 繪製、音效、DOM 更新全部
// 交纏在 1900 行的單一檔案裡，用 ~40 個模組級全域變數當狀態，
// 導致「無法單元測試、無法離線回放」——於是所有調參都只能盲調。
//
// 這裡的規則：
//   * 狀態全部掛在 this 上，可以被完整檢視與重置
//   * 不 import 任何 DOM 模組；影像來源只要是「可被 canvas drawImage 的東西」
//     （HTMLVideoElement / ImageBitmap / OffscreenCanvas 都符合）
//   * tick() 回傳 events 與 hud 資料，由呼叫端決定怎麼呈現
//   → 於是同一份 pipeline 可以同時被「即時 App」與「離線回放評測」驅動

import { CONFIG } from '../config.js';
import { Tracker } from '../tracking/tracker.js';
import { OpticalFlow } from '../motion/opticalFlow.js';
import { DepartureDetector } from '../motion/departure.js';
import { EgoMotionEstimator, EgoState } from '../motion/egoMotion.js';
import { FrontCarSelector } from '../logic/frontCar.js';
import { TrafficLightDetector } from '../logic/trafficLight.js';

export class Pipeline {
  constructor({ cfg = CONFIG, gps = null, imu = null } = {}) {
    this.cfg = cfg;
    this.gps = gps;
    this.imu = imu;

    this.tracker = new Tracker(cfg);
    this.flow = new OpticalFlow(cfg);
    this.departure = new DepartureDetector(cfg);
    this.ego = new EgoMotionEstimator(cfg, gps, imu);
    this.frontCar = new FrontCarSelector(cfg);
    this.light = new TrafficLightDetector(cfg);

    this.enableCarDepart = true;
    this.enableTrafficLight = true;

    this.lastFlow = null;
    this.lastFlowTs = 0;
    this.lastFlowMs = 0;
    this.lastLightTs = 0;
    this.lastTickTs = 0;
    this.visBgResid = null;
    this.imuDisagree = 0;
    this.trusted = true;
    this.target = null;
    this.stats = { ticks: 0, flowOk: 0, flowFail: {}, detections: 0 };
  }

  async initCv(loadScript) {
    return this.flow.loadCv(loadScript);
  }

  reset() {
    this.tracker.reset();
    this.flow.reset();
    this.departure.reset();
    this.ego.reset();
    this.frontCar.reset();
    this.light.reset();
    this.lastFlow = null;
    this.lastFlowTs = 0;
    this.lastTickTs = 0;
    this.visBgResid = null;
    this.target = null;
    this.stats = { ticks: 0, flowOk: 0, flowFail: {}, detections: 0 };
  }

  /** 偵測結果抵達（帶著它自己的影像時間戳，可能已延遲數百毫秒） */
  onDetections(boxes, ts) {
    this.stats.detections++;
    this.tracker.update(boxes, ts);
  }

  setFeatures({ car, light }) {
    if (car !== undefined) {
      this.enableCarDepart = car;
      if (!car) { this.departure.reset(); this.flow.reset(); this.frontCar.reset(); }
    }
    if (light !== undefined) {
      this.enableTrafficLight = light;
      if (!light) this.light.reset();
    }
  }

  /**
   * 每個 video frame 呼叫一次。
   * @returns { events: [{type, text}], hud: {...} }
   */
  tick({ source, vw, vh, now }) {
    const cfg = this.cfg;
    const events = [];
    this.stats.ticks++;

    // ---- 1) 地平線（IMU 重力向量）----
    // 注意單位：RLS 學到的增益是「rad → ROI 像素」，因為它是用 ROI 座標的
    // 背景位移訓練的。要用在整張影像上，必須除以 roiScale 換回影像像素。
    if (this.imu) {
      const fRoi = this.imu.focalPxEstimate();
      if (fRoi) {
        const roiScale = this.flow.roiScale > 0 ? this.flow.roiScale : 1;
        const fVideo = fRoi / roiScale;
        const hr = this.imu.horizonRatio(fVideo, vh);
        if (hr !== null) this.frontCar.setHorizon(hr);
      }
    }

    // ---- 2) 自車運動（三路融合）----
    this.ego.update(now, this.visBgResid);
    const canAlert = this.ego.canAlert;

    // ---- 3) 目標選取 ----
    const vehTracks = this.tracker.confirmedOf(cfg.vehicleClasses, now);
    const lightTracks = this.tracker.confirmedOf([cfg.lightClass], now);

    let target = null;
    if (this.enableCarDepart) {
      target = this.frontCar.select(vehTracks, vw, vh, now);
    }
    const targetChanged = target && this.target && target.id !== this.target.id;
    if (targetChanged) {
      // 換了目標 → 之前累積的證據對新目標無意義
      this.departure.reset();
      this.flow.reset();
    }
    this.target = target;

    // ---- 4) 光流 + 起步判定 ----
    let flowRes = null;
    if (this.enableCarDepart && target && this.flow.cvReady) {
      const minGap = 1000 / cfg.loop.flowHz;
      if (now - this.lastFlowTs >= minGap) {
        const box = target.boxAt(now);      // KF 預測到「現在」，不是幾百毫秒前的舊框
        const prevTs = this.lastFlowTs || now;
        const tf0 = performance.now();
        flowRes = this.flow.measure(source, box, vw, vh, now);
        this.lastFlowMs = performance.now() - tf0;
        this.lastFlowTs = now;
        this.lastFlow = flowRes;

        if (flowRes.ok) {
          this.stats.flowOk++;
          this._crossCheckImu(flowRes, prevTs, now);
          const r = this.departure.update(flowRes, {
            ts: now,
            egoStill: canAlert,
            trusted: this.trusted,
          });
          if (r.fired) events.push({ type: 'move', text: '🚗 前車已起步！', kind: 'depart' });
        } else {
          this.stats.flowFail[flowRes.reason] = (this.stats.flowFail[flowRes.reason] || 0) + 1;
          this.departure.coast(now);
        }
      }
    } else if (this.enableCarDepart) {
      // 沒有目標：coast 而非硬重置，短暫漏檢不該清空證據
      if (this.departure.lastMeasTs) this.departure.coast(now);
      else this.departure.reset();
      if (!target) this.flow.reset();
    }

    // ---- 5) 紅綠燈 ----
    let lightInfo = null;
    if (this.enableTrafficLight) {
      const minGap = 1000 / cfg.loop.lightHz;
      if (now - this.lastLightTs >= minGap) {
        this.lastLightTs = now;
        lightInfo = this.light.update(source, lightTracks, vw, vh, now, canAlert);
        if (lightInfo.fired) events.push({ type: 'green', text: '🟢 綠燈了！起步！', kind: 'green' });
      }
    }

    this.lastTickTs = now;

    return {
      events,
      hud: {
        ego: this.ego.state,
        egoSource: this.ego.source,
        egoLabel: this.ego.label,
        canAlert,
        speed: this.ego.speed,
        target: target ? { id: target.id, box: target.boxAt(now), score: target.score } : null,
        tracks: this.tracker.tracks.map((t) => ({
          id: t.id, classId: t.classId, box: t.boxAt(now),
          confirmed: t.confirmed, score: t.score,
          coasting: now - t.lastSeenTs > 60,
        })),
        horizon: this.frontCar.horizon,
        laneCenterX: this.frontCar.laneCenterX,
        corridor: this._corridorPoly(vw, vh),
        departure: this.departure.status(now),
        flow: flowRes,
        lastFlow: this.lastFlow,
        light: lightInfo,
        lightColor: this.light.lastColor,
        trusted: this.trusted,
        imuDisagree: this.imuDisagree,
        imuCalibrated: this.imu ? this.imu.calibrated : false,
        imuQuality: this.imu ? this.imu.calibX.quality : 0,
        stats: this.stats,
      },
    };
  }

  /**
   * IMU 交叉檢核：
   *   1. 把「陀螺儀積分出的旋轉角」對「視覺觀測到的背景位移」做線上回歸，
   *      學出增益（含符號、焦距、安裝姿態）——不需要手填 FOV
   *   2. 增益學好後，反過來用陀螺儀預測背景該怎麼動。
   *      若預測與觀測嚴重不符，代表「背景」本身不可信
   *      （最常見的原因就是旁車道車流佔了背景點多數，RANSAC 鎖到移動車群
   *       —— 這正是 v6 誤報起步的主要來源之一）
   *      → 本 tick 標記為不可信，不累積證據
   *   3. 扣掉旋轉後的殘差就是相機的平移分量，交給 egoMotion 當靜止判定的備援
   */
  _crossCheckImu(flowRes, prevTs, now) {
    this.trusted = true;
    this.imuDisagree = 0;
    this.visBgResid = null;
    if (!this.imu || !this.imu.hasGyro) return;

    const rot = this.imu.integrateRotation(prevTs, now);
    const obs = flowRes.bgObs;

    if (this.imu.calibrated) {
      const pred = this.imu.predictImageMotion(rot);
      const ex = obs.dx - pred.dx;
      const ey = obs.dy - pred.dy;
      const residMag = Math.hypot(ex, ey);
      const sigma = Math.hypot(this.imu.calibX.residSigma, this.imu.calibY.residSigma);
      if (sigma > 1e-6) {
        this.imuDisagree = residMag / sigma;
        if (this.imuDisagree > this.cfg.imu.disagreeSigma) this.trusted = false;
      }
      // 扣掉旋轉後的平移殘差（換回影像 px）→ 自車平移的視覺證據
      this.visBgResid = residMag / Math.max(flowRes.roi.scale, 1e-3);
    }

    // 學習永遠要做（包含尚未校準完成時）
    this.imu.teachImageMotion(rot, obs);
  }

  _corridorPoly(vw, vh) {
    const fc = this.frontCar;
    const pts = [];
    for (let i = 0; i <= 8; i++) {
      const yr = fc.horizon + (1 - fc.horizon) * (i / 8);
      const hw = fc.halfWidthAt(yr, vh / vw);
      pts.push({ yr, left: fc.laneCenterX - hw, right: fc.laneCenterX + hw });
    }
    return pts;
  }

  debugLines() {
    const l = [];
    l.push(this.ego.debugLine());
    l.push(this.frontCar.debugLine());
    l.push(this.departure.debugLine());
    if (this.lastFlow) {
      const f = this.lastFlow;
      l.push(f.ok
        ? `flow fg=${f.fg.nIn}/${f.fg.n} bg=${f.bg.nIn}/${f.bg.n}(${(f.bg.inlierRatio * 100).toFixed(0)}%)`
          + ` sRel=${f.sRel.toFixed(5)}±${f.sigmaRel.toExponential(1)} dy=${f.dyRel.toFixed(2)}`
        : `flow FAIL: ${f.reason}`);
    }
    l.push(`imu cal=${this.imu?.calibrated ? 'Y' : 'N'} q=${(this.imu?.calibX.quality || 0).toFixed(2)}`
      + ` disagree=${this.imuDisagree.toFixed(1)}σ trusted=${this.trusted ? 'Y' : 'N'}`);
    l.push(this.light.debugLine());
    return l;
  }
}

export { EgoState };
