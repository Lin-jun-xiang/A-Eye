// =============================================
// 影像來源 + 幀迴圈
// =============================================
// v6 用 setTimeout(detect, 100) 假設固定 100ms 節拍，但實際 tick 完全被
// 推論耗時支配。這裡改成跟著 video 的真實幀事件跑：
//   requestVideoFrameCallback 可用時用它（每個解碼出來的幀恰好觸發一次，
//   而且會帶 mediaTime，不會重複處理同一幀），否則退回 requestAnimationFrame。

export class FrameSource {
  constructor(videoEl) {
    this.video = videoEl;
    this.stream = null;
    this.fileUrl = null;
    this.isFile = false;
    this.running = false;
    this.onFrame = null;
    this._handle = null;
    this.usesRvfc = typeof videoEl.requestVideoFrameCallback === 'function';
  }

  async startCamera({ width, height }) {
    if (this.isFile) this.stopFile();      // 從影片模式切回相機
    this.video.loop = false;
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: width },
        height: { ideal: height },
      },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    // 等到真的有尺寸再回來，否則第一幀的 vw/vh 是 0
    if (!this.video.videoWidth) {
      await new Promise((res) => {
        const t = setTimeout(res, 3000);
        this.video.addEventListener('loadedmetadata', () => { clearTimeout(t); res(); }, { once: true });
      });
    }
    return { vw: this.video.videoWidth, vh: this.video.videoHeight };
  }

  /**
   * 用影片檔取代相機。
   * 為什麼要有這個：實車路測一趟只能驗一次，而同一段影片可以在每次改動後
   * 重跑 —— 這是把「盲調」變成「可重現」的關鍵。與 replay.html 的差別是
   * 這裡走的是即時路徑（rVFC、掉幀、真實節拍），看到的就是手機上的行為；
   * replay.html 走的是逐格 seek 的決定性路徑，用來算客觀指標。
   */
  async startFile(file) {
    this.stopCamera();
    if (this.fileUrl) URL.revokeObjectURL(this.fileUrl);
    this.fileUrl = URL.createObjectURL(file);
    this.video.srcObject = null;
    this.video.src = this.fileUrl;
    this.video.loop = true;          // 循環播放，方便反覆看同一個起步瞬間
    this.video.muted = true;
    this.isFile = true;
    await new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error('影片載入逾時')), 15000);
      this.video.onloadedmetadata = () => { clearTimeout(t); res(); };
      this.video.onerror = () => { clearTimeout(t); rej(new Error('此影片無法解碼（iOS 上 webm 常見）')); };
    });
    await this.video.play();
    return { vw: this.video.videoWidth, vh: this.video.videoHeight };
  }

  stopFile() {
    this.isFile = false;
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    if (this.fileUrl) { URL.revokeObjectURL(this.fileUrl); this.fileUrl = null; }
  }

  stopCamera() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  /** 相機的實際設定（用於除錯：解析度往往不等於 ideal） */
  settings() {
    const t = this.stream && this.stream.getVideoTracks()[0];
    return t ? t.getSettings() : null;
  }

  start(onFrame) {
    this.onFrame = onFrame;
    this.running = true;
    this._schedule();
  }

  stop() {
    this.running = false;
    if (this._handle !== null) {
      if (this.usesRvfc && this.video.cancelVideoFrameCallback) {
        this.video.cancelVideoFrameCallback(this._handle);
      } else {
        cancelAnimationFrame(this._handle);
      }
      this._handle = null;
    }
  }

  _schedule() {
    if (!this.running) return;
    if (this.usesRvfc) {
      this._handle = this.video.requestVideoFrameCallback((now, meta) => {
        this._handle = null;
        if (!this.running) return;
        try { this.onFrame(performance.now(), meta); } catch (e) { console.error('[A-Eye] frame error:', e); }
        this._schedule();
      });
    } else {
      this._handle = requestAnimationFrame(() => {
        this._handle = null;
        if (!this.running) return;
        try { this.onFrame(performance.now(), null); } catch (e) { console.error('[A-Eye] frame error:', e); }
        this._schedule();
      });
    }
  }
}
