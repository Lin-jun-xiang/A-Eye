// =============================================
// 錄影 + 感測器記錄（離線評測的資料來源）
// =============================================
// 這是「階段 1」：沒有可回放的資料集，任何調參都是盲調。
// 錄下三樣東西：
//   1. 影片（webm）
//   2. 感測器軌跡（JSONL：GPS 速度、IMU 陀螺儀/加速度）帶上與影片同源的時間戳
//   3. 事件標註（使用者在錄影時按一下「前車起步了」按鈕即可打點）
// 之後用 replay.html 餵回同一個 pipeline，就能算出 FP/小時、漏報率、偵測延遲。

export class SessionRecorder {
  constructor({ video, gps, imu }) {
    this.video = video;
    this.gps = gps;
    this.imu = imu;
    this.recorder = null;
    this.chunks = [];
    this.sensorLog = [];
    this.marks = [];
    this.startTs = 0;
    this.recording = false;
    this._timer = null;
    this._imuIdx = 0;
  }

  get supported() {
    return typeof MediaRecorder !== 'undefined' && !!this.video.srcObject;
  }

  start() {
    if (this.recording || !this.supported) return false;
    const stream = this.video.srcObject;
    const types = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4',
    ];
    const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t));
    try {
      this.recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 4e6 } : undefined);
    } catch (e) {
      console.warn('[A-Eye] MediaRecorder 建立失敗:', e.message);
      return false;
    }
    this.chunks = [];
    this.sensorLog = [];
    this.marks = [];
    this._imuIdx = 0;
    this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.startTs = performance.now();
    this.recorder.start(1000);
    this.recording = true;

    // 感測器以 20Hz 取樣寫入（IMU 原始資料太密，取樣後仍足夠重建）
    this._timer = setInterval(() => this._sampleSensors(), 50);
    return true;
  }

  _sampleSensors() {
    const t = performance.now() - this.startTs;
    const rec = { t: Math.round(t) };
    if (this.gps) {
      rec.gpsSpeed = this.gps.speed;
      rec.gpsAge = this.gps.ts ? Math.round(performance.now() - this.gps.ts) : null;
    }
    if (this.imu && this.imu.samples.length) {
      const s = this.imu.samples[this.imu.samples.length - 1];
      rec.gyro = [+s.rx.toFixed(5), +s.ry.toFixed(5), +s.rz.toFixed(5)];
      rec.accel = [+s.ax.toFixed(4), +s.ay.toFixed(4), +s.az.toFixed(4)];
    }
    this.sensorLog.push(rec);
  }

  /** 使用者在現場打的事件標註（ground truth） */
  mark(label) {
    if (!this.recording) return null;
    const m = { t: Math.round(performance.now() - this.startTs), label };
    this.marks.push(m);
    return m;
  }

  async stop() {
    if (!this.recording) return null;
    clearInterval(this._timer);
    this._timer = null;
    const done = new Promise((res) => { this.recorder.onstop = res; });
    this.recorder.stop();
    await done;
    this.recording = false;

    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const blob = new Blob(this.chunks, { type: this.recorder.mimeType || 'video/webm' });
    const ext = (this.recorder.mimeType || '').includes('mp4') ? 'mp4' : 'webm';

    const meta = {
      recordedAt: new Date().toISOString(),
      durationMs: Math.round(performance.now() - this.startTs),
      videoWidth: this.video.videoWidth,
      videoHeight: this.video.videoHeight,
      userAgent: navigator.userAgent,
      marks: this.marks,
    };

    this._download(blob, `aeye-${stamp}.${ext}`);
    this._download(
      new Blob([this.sensorLog.map((r) => JSON.stringify(r)).join('\n')], { type: 'application/x-ndjson' }),
      `aeye-${stamp}.sensors.jsonl`
    );
    this._download(
      new Blob([JSON.stringify(meta, null, 2)], { type: 'application/json' }),
      `aeye-${stamp}.meta.json`
    );

    return { meta, sizeBytes: blob.size, marks: this.marks.length };
  }

  _download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
}
