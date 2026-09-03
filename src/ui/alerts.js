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
    }
  }

  vibrate(kind) {
    if (!navigator.vibrate) return;
    navigator.vibrate(kind === 'depart' ? [120, 60, 120, 60, 200] : [200, 80, 200]);
  }

  flash(kind) {
    const el = this.flashEl;
    if (!el) return;
    el.className = kind === 'depart' ? 'move-flash active' : 'green-flash active';
    setTimeout(() => { el.className = ''; }, 500);
  }

  fire(event) {
    this.sound(event.kind);
    this.vibrate(event.kind);
    this.flash(event.kind);
  }

  render(badges) {
    if (!this.alertsEl) return;
    this.alertsEl.innerHTML = badges.map(
      (b) => `<div class="alert-badge ${b.type}">${b.text}</div>`
    ).join('');
  }
}
