// =============================================
// 警示輸出（音效 / 震動 / 閃爍 / 徽章）
// =============================================

export class AlertPresenter {
  constructor({ alertsEl, flashEl }) {
    this.alertsEl = alertsEl;
    this.flashEl = flashEl;
    this.audioCtx = null;
  }

  ensureAudio() {
    if (!this.audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) this.audioCtx = new AC();
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') this.audioCtx.resume();
    return this.audioCtx;
  }

  beep(freq, dur, vol = 0.35, delay = 0) {
    const ctx = this.audioCtx;
    if (!ctx) return;
    const t0 = ctx.currentTime + delay;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = freq;
    o.type = 'sine';
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(ctx.destination);
    o.start(t0); o.stop(t0 + dur + 0.02);
  }

  sound(kind) {
    this.ensureAudio();
    if (kind === 'depart') {
      this.beep(880, 0.12, 0.35, 0);
      this.beep(1180, 0.12, 0.35, 0.14);
      this.beep(1480, 0.18, 0.35, 0.28);
    } else if (kind === 'green') {
      this.beep(660, 0.16, 0.32, 0);
      this.beep(990, 0.24, 0.32, 0.18);
    } else if (kind === 'release') {
      // 「前車鬆開剎車」是預告不是事件 —— 刻意做得比正式警示輕：
      // 單音、音量減半。若做得一樣響，駕駛會分不出「該動了」和「快要該動了」。
      this.beep(760, 0.10, 0.18, 0);
    }
  }

  vibrate(kind) {
    if (!navigator.vibrate) return;
    const pattern = kind === 'depart' ? [120, 60, 120, 60, 200]
      : kind === 'release' ? [90]
      : [200, 80, 200];
    navigator.vibrate(pattern);
  }

  flash(kind) {
    const el = this.flashEl;
    if (!el) return;
    if (kind === 'release') return;      // 預告不閃全螢幕
    el.className = kind === 'depart' ? 'move-flash active' : 'green-flash active';
    setTimeout(() => { el.className = ''; }, 500);
  }

  fire(event) {
    // 無聲事件：只在畫面上留下痕跡，不出聲、不震動、不閃。
    // 「前車踩下剎車」領先約 9 秒 —— 那麼早出聲只會變成干擾，
    // 但它仍然是有價值的資訊（尤其在離線分析時間軸上）。
    if (event.silent) return;
    this.sound(event.kind);
    this.vibrate(event.kind);
    this.flash(event.kind);
  }

  render(badges) {
    if (!this.alertsEl) return;
    const html = badges.map(
      (b) => `<div class="alert-badge ${b.type}">${b.text}</div>`
    ).join('');
    // 內容沒變就不寫 DOM。重建 innerHTML 會讓瀏覽器丟掉整棵子樹再重排，
    // 這才是「徽章一直閃、讀不出字」的直接來源 —— 呼叫端限流還不夠，
    // 因為即使限到 5Hz，每次都重建 DOM 一樣會閃。
    if (html === this._lastHtml) return;
    this._lastHtml = html;
    this.alertsEl.innerHTML = html;
  }
}
