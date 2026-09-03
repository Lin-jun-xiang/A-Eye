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
    this.running = false;
    this.onFrame = null;
    this._handle = null;
    this.usesRvfc = typeof videoEl.requestVideoFrameCallback === 'function';
  }

  async startCamera({ width, height }) {
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
