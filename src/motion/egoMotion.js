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
//
// IMU 的「靜止基線」不寫死，而是在 GPS 確認靜止時線上學習。

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
  }

  reset() {
    this.state = EgoState.UNKNOWN;
    this.source = 'none';
    this.speed = null;
    this._visResidSigma = null;
  }

  /**
   * 每 tick 呼叫。
   * @param visBgResidPx 背景位移在扣掉陀螺儀預測後的殘差大小（px），無則傳 null
   */
  update(now, visBgResidPx = null) {
    const e = this.cfg.ego;

    // ---- 1) GPS（主）----
    const sp = this.gps ? this.gps.currentSpeed(now) : null;
    this.speed = sp;
    if (sp !== null) {
      const wasMoving = this.state === EgoState.MOVING;
      const moving = wasMoving ? sp > e.gpsStillSpeed : sp > e.gpsMoveSpeed;
      this.state = moving ? EgoState.MOVING : EgoState.STILL;
      this.source = 'gps';
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
    if (visBgResidPx !== null && isFinite(visBgResidPx)) {
      this._lastVisResid = visBgResidPx;
      if (this._visResidSigma === null) {
        this._visResidSigma = Math.max(visBgResidPx, 0.2);
      } else {
        // 用穩健的方式追蹤雜訊尺度：只在「小值」上更新，避免被真實運動污染
        if (visBgResidPx < this._visResidSigma * 2) {
          this._visResidSigma = 0.97 * this._visResidSigma + 0.03 * Math.max(visBgResidPx, 0.05);
        }
      }
      const moving = visBgResidPx > this._visResidSigma * e.visualMoveSigma;
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
      + (this._visResidSigma !== null ? ` visσ=${this._visResidSigma.toFixed(2)}` : '');
  }
}
