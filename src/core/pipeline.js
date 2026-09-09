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
import { iou } from '../util/math.js';
import { Tracker } from '../tracking/tracker.js';
import { OpticalFlow } from '../motion/opticalFlow.js';
import { DepartureDetector } from '../motion/departure.js';
import { EgoMotionEstimator, EgoState } from '../motion/egoMotion.js';
import { FrontCarSelector } from '../logic/frontCar.js';
import { TrafficLightDetector } from '../logic/trafficLight.js';
import { BrakeLightDetector } from '../logic/brakeLight.js';

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
    this.brake = new BrakeLightDetector(cfg);
    this.light = new TrafficLightDetector(cfg);

    this.enableCarDepart = true;
    this.enableTrafficLight = true;
    // 影片模式：檔案沒有 GPS / IMU，自車是否靜止無從得知。
    // 這是一個「明確宣告的假設」而不是靜默的行為改變 —— hud 會把它傳給 UI 顯示。
    this.assumeStill = false;

    this.lastFlow = null;
    this.lastFlowTs = 0;
    this.lastFlowMs = 0;
    this.lastLightTs = 0;
    this.lastBrakeTs = 0;
    // trackId -> { ms, everOn }：某個框被觀察了多久、期間有沒有出現過剎車燈。
    // 「觀察夠久卻始終沒有一對對稱的紅燈」是自車結構/反光的正向反證。
    this.lampEvidence = new Map();
    this.lastTickTs = 0;
    // 給 egoMotion 的視覺證據（上一個 tick 算出來的，因為 ego 判定在光流之前）
    this.visEvidence = null;
    this.imuDisagree = 0;
    this.trusted = true;
    this.target = null;
    this.lastTargetBox = null;
    this.lastTargetTs = 0;
    this.stats = {
      ticks: 0, flowOk: 0, flowFail: {}, detections: 0,
      targetTicks: 0, targetFresh: 0, targetChanges: 0, evidenceResets: 0,
    };
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
    this.brake.reset();
    this.light.reset();
    this.lampEvidence.clear();
    this.lastBrakeTs = 0;
    this.lastFlow = null;
    this.lastFlowTs = 0;
    this.lastTickTs = 0;
    this.visEvidence = null;
    this.target = null;
    this.lastTargetBox = null;
    this.lastTargetTs = 0;
    this.stats = {
      ticks: 0, flowOk: 0, flowFail: {}, detections: 0,
      targetTicks: 0, targetFresh: 0, targetChanges: 0, evidenceResets: 0,
    };
  }

  /** 偵測結果抵達（帶著它自己的影像時間戳，可能已延遲數百毫秒） */
  onDetections(boxes, ts) {
    this.stats.detections++;
    this.tracker.update(boxes, ts);
  }

  setFeatures({ car, light }) {
    if (car !== undefined) {
      this.enableCarDepart = car;
      if (!car) {
        this.departure.reset(); this.flow.reset(); this.frontCar.reset();
        this.brake.reset(); this.lampEvidence.clear();
      }
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
    this.ego.update(now, this.visEvidence);
    const canAlert = this.ego.canAlert || this.assumeStill;

    // ---- 3) 目標選取 ----
    const vehTracks = this.tracker.confirmedOf(cfg.vehicleClasses, now);
    const lightTracks = this.tracker.confirmedOf([cfg.lightClass], now);

    let target = null;
    if (this.enableCarDepart) {
      // 先學自車結構（引擎蓋 / 儀表板 / 反光被 YOLO 認成 car）。
      // 它的底邊在畫面最下方 → proximity 最高 → 不排除就會永遠被選成「前車」，
      // 於是光流量的是一個永遠不動的東西，起步警示結構上不可能觸發。
      this.frontCar.learnEgoStructure(
        vehTracks, vw, vh, now, this.ego.state === EgoState.MOVING
      );
      target = this.frontCar.select(vehTracks, vw, vh, now, {
        lampWeightFn: (id) => this._lampWeight(id),
      });
    }
    // ---- 換目標時要不要清空證據 ----
    // 原本是「id 一變就清空」，但實車量測顯示這會在最需要警示的那一刻把證據丟掉：
    // 夜間近距離的白車，YOLO 只有 21% 的畫格抓得到（76% 在 coasting），
    // track 因此反覆被淘汰重建 —— 104 秒內換了 28 個 id。
    // 每次換 id 就 reset，證據永遠從零開始（實測面板：證據 0%、reason=idle）。
    //
    // 但「id 變了」不等於「換了一台車」：track 死掉後在同一個位置被重建，
    // 幾何上仍是同一台車。所以改用「新舊目標框的 IoU」判斷是不是同一個物體 ——
    // 這是可觀測的幾何事實，比 track id 這個實作細節可靠。
    const newBox = target ? target.boxAt(now) : null;
    if (target && this.target && target.id !== this.target.id) {
      const same = this.lastTargetBox && iou(this.lastTargetBox, newBox) >= cfg.tracker.sameTargetIou;
      this.stats.targetChanges++;
      if (!same) {
        this.stats.evidenceResets++;
        this.departure.reset();
        this.flow.reset();
        this.brake.reset();
      }
      // 幾何上是同一台車 → 保留證據。光流的錨定 ROI 本來就會在目標
      // 跑出錨定框時自己重新錨定，不需要在這裡硬清。
    }
    this.target = target;
    this.lastTargetBox = newBox;
    if (target) this.lastTargetTs = now;
    // 目標的「新鮮度」：這個 tick 的目標有沒有剛被偵測更新過。
    // 這個比例就是上面那條因果鏈的源頭，必須看得見。
    if (target) {
      this.stats.targetTicks++;
      if (now - target.lastSeenTs <= 1.8 * (1000 / cfg.loop.detectHz)) this.stats.targetFresh++;
    }

    // ---- 3.5) 剎車燈（快路徑）----
    // 放在光流之前：它的結果會當成起步判定的先驗，而且比運動訊號早 0.3~1 秒。
    let brakeRes = null;
    if (this.enableCarDepart && target) {
      const gap = 1000 / cfg.brakeLight.hz;
      if (now - this.lastBrakeTs >= gap) {
        const dtMs = this.lastBrakeTs ? now - this.lastBrakeTs : gap;
        this.lastBrakeTs = now;
        brakeRes = this.brake.update(source, target.boxAt(now), now);
        this._noteLamp(target.id, brakeRes, dtMs);
        const armed = now - target.firstTs >= cfg.departure.armMs;
        if (brakeRes.released && cfg.brakeLight.alertOnRelease && canAlert && armed) {
          events.push({ type: 'release', text: '🟠 前車鬆開剎車，準備起步', kind: 'release' });
        }
        // 「踩下」：自排車從 P/N 打入 D 必須踩剎車，所以「暗了很久之後亮起」
        // 是起步的前兆 —— 實車量測領先約 9 秒（38.3s 踩下 → 47.4s 車動）。
        // 但 9 秒太早，出聲會變成干擾，所以預設是**無聲事件**：
        // 徽章與分析時間軸看得到，不發出聲音與震動。
        if (brakeRes.pressed && canAlert && armed) {
          events.push({
            type: 'ready', kind: 'ready',
            text: '⏸ 前車踩下剎車（可能正在打檔）',
            silent: !cfg.brakeLight.chmsl.alertOnPress,
          });
        }
      }
    }

    // ---- 4) 光流 + 起步判定 ----
    let flowRes = null;
    if (this.enableCarDepart && target && this.flow.cvReady) {
      const minGap = 1000 / cfg.loop.flowHz;
      if (now - this.lastFlowTs >= minGap) {
        const box = target.boxAt(now);      // KF 預測到「現在」，不是幾百毫秒前的舊框
        const tf0 = performance.now();
        flowRes = this.flow.measure(source, box, vw, vh, now);
        this.lastFlowMs = performance.now() - tf0;
        this.lastFlowTs = now;
        this.lastFlow = flowRes;

        if (flowRes.ok) {
          this.stats.flowOk++;
          // 用量測自己的時間區間（不是「距上一幀」）積分陀螺儀，
          // 才能和背景位移對得起來
          this._crossCheckImu(flowRes, flowRes.t0, flowRes.t1);
          const r = this.departure.update(flowRes, {
            ts: now,
            egoStill: canAlert,
            trusted: this.trusted,
            primed: this.brake.primed(now),
            priorLlr: this.brake.priorLlr(),
            armed: now - target.firstTs >= cfg.departure.armMs,
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
      // 這裡原本每個 tick 都 flow.reset()，代價很大：光流的特徵點與錨定 ROI
      // 被立刻銷毀，目標一回來就得從零重新錨定、重新累積 240ms 基線。
      // 而實車量測顯示目標「短暫消失」是常態（79% 的 tick 目標都不是新的）。
      // 改成只有超過追蹤器的 coast 時間才真的放棄 —— 在那之前保留錨定，
      // 目標回來時若還在錨定框內就能直接接續。
      if (!target && this.lastTargetTs && now - this.lastTargetTs > cfg.tracker.maxCoastMs) {
        this.flow.reset();
      }
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
        bgExpZ: this.visEvidence ? this.visEvidence.bgExpZ : null,
        egoLabel: this.ego.label,
        canAlert,
        assumeStill: this.assumeStill,
        speed: this.ego.speed,
        target: target ? { id: target.id, box: target.boxAt(now), score: target.score } : null,
        tracks: this.tracker.tracks.map((t) => ({
          id: t.id, classId: t.classId, box: t.boxAt(now),
          confirmed: t.confirmed, score: t.score,
          // 「還沒被新的偵測更新」的門檻要跟偵測週期綁在一起。寫死 60ms 的話，
          // detectHz=8（125ms 一次）會讓每個框幾乎永遠是虛線。
          coasting: now - t.lastSeenTs > 1.8 * (1000 / cfg.loop.detectHz),
        })),
        horizon: this.frontCar.horizon,
        laneCenterX: this.frontCar.laneCenterX,
        egoRegions: this.frontCar.egoRegions,
        corridor: this._corridorPoly(vw, vh),
        departure: this.departure.status(now),
        brake: brakeRes,
        brakeState: this.brake.state,
        brakePrimed: this.brake.primed(now),
        brakeDetail: this.brake.lastDetail,
        brakeCfg: cfg.brakeLight,
        // 判定「熄滅」靠的是「位準 / 峰值」這個比值，所以峰值與門檻要一起給 UI
        // 第三剎車燈：可用時它就是權威判據（落差 70 倍 vs 外側燈的 1.5 倍）
        brakeChmsl: brakeRes && brakeRes.chmsl ? brakeRes.chmsl : null,
        brakeChmslState: this.brake.chState,
        brakeChmslUsable: this.brake.chUsable,
        brakeChmslVal: this.brake.chVal,
        brakeChmslPeak: this.brake.chPeak,
        brakeChmslFar: this.brake.chFar,
        brakeChmslFound: !!this.brake.chPos,
        brakePressed: this.brake.pressed(now),
        brakePeak: Math.min(this.brake.peakL, this.brake.peakR),
        brakeLevel: Math.min(this.brake.levelL, this.brake.levelR),
        brakeBlinking: this.brake.blinking,
        brakeOffRatio: cfg.brakeLight.offRatio,
        flow: flowRes,
        lastFlow: this.lastFlow,
        light: lightInfo,
        lightColor: this.light.lastColor,
        trusted: this.trusted,
        imuDisagree: this.imuDisagree,
        imuCalibrated: this.imu ? this.imu.calibrated : false,
        imuQuality: this.imu ? this.imu.calibX.quality : 0,
        stats: this.stats,
        trackerStats: this.tracker.stats,
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

    // ---- 背景尺度變化率 → 自車是否在前進 ----
    // 自車前進 → 背景（路面、建物、旁車）逼近 → 背景尺度 > 1。
    // 這是判斷「自車有沒有在動」最直接的視覺證據，而且**旋轉不改變尺度**，
    // 所以它對手機晃動天然免疫（位移殘差那一路不是）。
    // 用途：否決 GPS 的靜止漂移（實測停在路口時 GPS 常跳 2~5 km/h）。
    const bg = flowRes.bg;
    const bgExpZ = (bg && bg.sigmaS > 0 && isFinite(bg.sigmaS))
      ? Math.log(bg.s) * bg.s / bg.sigmaS
      : null;
    this.visEvidence = { resid: null, bgExpZ };

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
      this.visEvidence.resid = residMag / Math.max(flowRes.roi.scale, 1e-3);
    }

    // 學習永遠要做（包含尚未校準完成時）
    this.imu.teachImageMotion(rot, obs);
  }

  /** 累積「這個框被觀察了多久、期間有沒有出現剎車燈」 */
  _noteLamp(id, res, dtMs) {
    if (!res || !res.usable) return;          // 過曝/裁切失敗不算證據
    let e = this.lampEvidence.get(id);
    if (!e) { e = { ms: 0, everOn: false }; this.lampEvidence.set(id, e); }
    e.ms += dtMs;
    if (res.everOn) e.everOn = true;
    // 只保留還活著的 track，避免長時間行駛後 Map 無限成長
    if (this.lampEvidence.size > 32) {
      const alive = new Set(this.tracker.tracks.map((t) => t.id));
      for (const k of this.lampEvidence.keys()) if (!alive.has(k)) this.lampEvidence.delete(k);
    }
  }

  /**
   * 剎車燈的正向證據加權：**只加分，不扣分**。
   * 只有被選中的目標會被分析，所以「扣分」會變成自我毀滅的迴路
   * （被選中者是唯一會被扣分的 → 一扣分就輸給沒被分析過的競爭者 → 震盪）。
   * 加分則讓同樣的不對稱變成穩定性：已確認有尾燈的目標更難被搶走。
   */
  _lampWeight(id) {
    const e = this.lampEvidence.get(id);
    return e && e.everOn ? this.cfg.frontCar.plausibility.lampBonus : 1;
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
    const st = this.stats;
    l.push(`target 新鮮 ${st.targetTicks ? (st.targetFresh / st.targetTicks * 100).toFixed(0) : '--'}%`
      + ` 換手 ${st.targetChanges} 次（其中清空證據 ${st.evidenceResets} 次）`);
    const ts2 = this.tracker.stats;
    l.push(`偵測框 ${ts2.dets} 個 → 配對 ${ts2.matched} 新建 ${ts2.created} 淘汰 ${ts2.dropped}`);
    l.push(this.departure.debugLine());
    l.push(this.brake.debugLine());
    if (this.lastFlow) {
      const f = this.lastFlow;
      l.push(f.ok
        ? `flow fg=${f.fg.nIn}/${f.fg.n} bg=${f.bg.nIn}/${f.bg.n}(${(f.bg.inlierRatio * 100).toFixed(0)}%)`
          + ` sRel=${f.sRel.toFixed(5)}±${f.sigmaRel.toExponential(1)} dy=${f.dyRel.toFixed(2)}`
        : `flow FAIL: ${f.reason}`);
    }
    const bz = this.visEvidence && this.visEvidence.bgExpZ;
    l.push(`bgExpZ=${bz === null || bz === undefined ? '--' : bz.toFixed(1)}`
      + `（|z|<${this.cfg.ego.visualStaticZ} → 自車視覺上靜止）`);
    l.push(`imu cal=${this.imu?.calibrated ? 'Y' : 'N'} q=${(this.imu?.calibX.quality || 0).toFixed(2)}`
      + ` disagree=${this.imuDisagree.toFixed(1)}σ trusted=${this.trusted ? 'Y' : 'N'}`);
    l.push(this.light.debugLine());
    return l;
  }
}

export { EgoState };
