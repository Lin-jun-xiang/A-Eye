// =============================================
// 自車運動狀態融合
// =============================================
// v6 只靠 GPS 的 pos.coords.speed，且「不可用時維持上一狀態」。
// 初值 egoMoving=false → GPS 一失效就等於行駛中也照報警；
// 反之若曾被判 moving 而 GPS 斷了，就永久靜默。兩種失效都很致命。
//
// 這裡改成三路融合，並且把「無法判斷」當成第三種明確狀態：
//   GPS speed      —— 主訊號，可用時優先
//   IMU 加速度變異數 —— GPS 為 null 時的即時備援（怠速停車與行駛特徵不同）
//   視覺背景位移    —— 最後備援（背景點在扣掉陀螺儀旋轉後幾乎不動 → 靜止）
//   視覺背景尺度    —— 自車前進 → 背景逼近 → 尺度 > 1。**旋轉不改變尺度**，
//                      所以這一路對手機晃動天然免疫，是低速下最可靠的判據
//
// IMU 的「靜止基線」不寫死，而是在 GPS 確認靜止時線上學習。
//
// v7.1 補的洞：GPS 的靜止漂移。實測停在路口時 coords.speed 會跳到 2~5 km/h，
// 而原本的遲滯不對稱（進 MOVING 只要一個尖峰、退出要掉到 0.5 m/s 以下）
// 會把狀態鎖在 MOVING → 整個紅燈都靜默，而且是**無聲的失效**：
// 畫面只顯示「行駛中 — 靜默中」，使用者不會知道自己其實已經停下來了。

export const EgoState = {
  STILL: 'still',
  MOVING: 'moving',
  UNKNOWN: 'unknown',
};

export class EgoMotionEstimator {
  constructor(cfg, gps, imu) {
    this.cfg = cfg;
    this.gps = gps;
    this.imu = imu;
    this.state = EgoState.UNKNOWN;
    this.source = 'none';
    this.speed = null;
    // 視覺備援用：背景殘差位移的雜訊尺度（線上學習）
    this._visResidSigma = null;
    this._lastVisResid = 0;
    this._aboveSince = 0;     // GPS 速度連續超過上緣的起始時刻
    this._vetoedTs = -Infinity;   // 最近一次用視覺否決 GPS 的時刻（除錯顯示用）
  }

  reset() {
    this.state = EgoState.UNKNOWN;
    this.source = 'none';
    this.speed = null;
    this._visResidSigma = null;
    this._aboveSince = 0;
    this._vetoedTs = -Infinity;
  }

  /**
   * 每 tick 呼叫。
   * @param vis 視覺證據，或 null。可以是舊介面的數字（背景殘差 px），
   *            或 { resid, bgExpZ }：
   *              resid  背景位移扣掉陀螺儀預測後的殘差大小（px）
   *              bgExpZ 背景尺度變化率的 z 值（>0 = 背景在逼近 = 自車前進）
   */
  update(now, vis = null) {
    const e = this.cfg.ego;
    const visResid = (vis && typeof vis === 'object') ? vis.resid
      : (typeof vis === 'number' ? vis : null);
    const bgExpZ = (vis && typeof vis === 'object') ? vis.bgExpZ : null;
    // 背景的尺度沒有顯著變化 → 沒有逼近任何東西 → 自車沒有前進。
    // 旋轉不改變尺度，所以這個判據對手機晃動天然免疫（不像位移殘差）。
    const visualStatic = bgExpZ !== null && bgExpZ !== undefined
      && isFinite(bgExpZ) && Math.abs(bgExpZ) < e.visualStaticZ;
    // 否決是間歇發生的（每次否決會把 dwell 計時歸零，所以下一個 tick 又從頭算），
    // 但除錯面板需要看得到「剛剛否決過」，否則這條路徑幾乎永遠不會被顯示出來
    const vetoRecent = now - this._vetoedTs < 2000;

    // ---- 1) GPS（主）----
    const sp = this.gps ? this.gps.currentSpeed(now) : null;
    const acc = this.gps ? this.gps.accuracy : null;
    this.speed = sp;
    // 定位精度太差時速度不可信 —— 都市高樓間 GPS 的速度是由位置差分來的，
    // 位置抖動幾十公尺就會憑空生出幾 km/h
    const gpsUsable = sp !== null
      && (acc === null || acc === undefined || acc <= e.gpsMaxAccuracyM);
    if (gpsUsable) {
      const wasMoving = this.state === EgoState.MOVING;
      // 進入 MOVING 需要「持續」超過上緣，而不是一個尖峰。
      // GPS 靜止漂移實測會跳到 2~5 km/h，偶爾一個尖峰就會把狀態鎖進 MOVING，
      // 而退出只需要掉到 0.5 m/s —— 這個不對稱會讓整個紅燈都靜默。
      if (sp > e.gpsMoveSpeed) {
        if (!this._aboveSince) this._aboveSince = now;
      } else {
        this._aboveSince = 0;
      }
      const sustained = this._aboveSince > 0 && now - this._aboveSince >= e.gpsMoveDwellMs;
      let moving = wasMoving ? sp > e.gpsStillSpeed : sustained;

      // 視覺否決：低速下 GPS 說在動、但背景完全沒有逼近 → 相信視覺
      if (moving && sp < e.gpsCreepCeiling && visualStatic) {
        moving = false;
        this._aboveSince = 0;
        this._vetoedTs = now;
      }
      this.state = moving ? EgoState.MOVING : EgoState.STILL;
      this.source = (vetoRecent || this._vetoedTs === now) ? 'gps+vis' : 'gps';
      // GPS 明確說靜止 → 拿來教 IMU 的靜止基線
      if (!moving && sp < e.gpsStillSpeed && this.imu) this.imu.teachStillBaseline();
      return this.state;
    }

    // ---- 2) IMU（備援）----
    if (this.imu) {
      const imuMoving = this.imu.isMovingByImu();
      if (imuMoving !== null) {
        this.state = imuMoving ? EgoState.MOVING : EgoState.STILL;
        this.source = 'imu';
        return this.state;
      }
    }

    // ---- 3) 視覺背景（最後備援）----
    // 背景在扣掉相機旋轉後仍有一致位移 → 自車在平移
    // 背景尺度率若可用，它比位移殘差更可靠（不受旋轉污染）
    if (bgExpZ !== null && bgExpZ !== undefined && isFinite(bgExpZ)) {
      this.state = visualStatic ? EgoState.STILL : EgoState.MOVING;
      this.source = 'visual-scale';
      return this.state;
    }
    if (visResid !== null && isFinite(visResid)) {
      this._lastVisResid = visResid;
      if (this._visResidSigma === null) {
        this._visResidSigma = Math.max(visResid, 0.2);
      } else {
        // 用穩健的方式追蹤雜訊尺度：只在「小值」上更新，避免被真實運動污染
        if (visResid < this._visResidSigma * 2) {
          this._visResidSigma = 0.97 * this._visResidSigma + 0.03 * Math.max(visResid, 0.05);
        }
      }
      const moving = visResid > this._visResidSigma * e.visualMoveSigma;
      this.state = moving ? EgoState.MOVING : EgoState.STILL;
      this.source = 'visual';
      return this.state;
    }

    // ---- 4) 全都不可用 ----
    this.state = EgoState.UNKNOWN;
    this.source = 'none';
    return this.state;
  }

  /** 是否可以發出「靜止時才有意義」的警示 */
  get canAlert() {
    if (this.state === EgoState.STILL) return true;
    if (this.state === EgoState.UNKNOWN) return this.cfg.ego.unknownIsStill;
    return false;
  }

  get label() {
    if (this.state === EgoState.MOVING) {
      return this.speed !== null ? `行駛中 ${Math.round(this.speed * 3.6)} km/h` : '行駛中';
    }
    if (this.state === EgoState.STILL) return '靜止';
    return '自車狀態未知';
  }

  debugLine() {
    return `ego=${this.state}(${this.source})`
      + (this.speed !== null ? ` ${(this.speed * 3.6).toFixed(1)}km/h` : '')
      + (this.gps && this.gps.accuracy ? ` acc=${this.gps.accuracy.toFixed(0)}m` : '')
      + (this.source === 'gps+vis' ? ' [視覺否決GPS漂移]' : '')
      + (this._visResidSigma !== null ? ` visσ=${this._visResidSigma.toFixed(2)}` : '');
  }
}
