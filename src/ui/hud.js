// =============================================
// 效能儀表 + 除錯面板
// =============================================
// 這是「階段 0」：v6 完全沒有任何耗時量測，所以沒人知道實際 tick 率是多少。
// 依 v6 的結構推估（每 tick 串行跑 YOLOv8s@640 + MiDaS@256 + UFLD@800x288）
// 實際大概只有 1~2 fps —— 而 LK 光流的 small-motion 假設在 dt=0.5~1s 下
// 根本不成立。沒有這個面板，任何調參都是盲調。

class Ema {
  constructor(a = 0.1) { this.a = a; this.v = null; }
  push(x) {
    if (!isFinite(x)) return;
    this.v = this.v === null ? x : (1 - this.a) * this.v + this.a * x;
  }
  get value() { return this.v; }
  str(d = 1) { return this.v === null ? '--' : this.v.toFixed(d); }
}

class RateMeter {
  constructor(windowMs = 2000) { this.windowMs = windowMs; this.ts = []; }
  mark(now) {
    this.ts.push(now);
    const cut = now - this.windowMs;
    while (this.ts.length && this.ts[0] < cut) this.ts.shift();
  }
  hz(now) {
    if (this.ts.length < 2) return 0;
    const span = (now - this.ts[0]) / 1000;
    return span > 0 ? (this.ts.length - 1) / span : 0;
  }
}

export class Metrics {
  constructor() {
    this.frameRate = new RateMeter();
    this.tickRate = new RateMeter();
    this.detectRate = new RateMeter();
    this.flowRate = new RateMeter();

    this.tickMs = new Ema();
    this.flowMs = new Ema();
    this.drawMs = new Ema();
    this.lightMs = new Ema();
    this.detPre = new Ema();
    this.detInfer = new Ema();
    this.detDecode = new Ema();
    this.detLatency = new Ema();
  }

  onDetectorResult(timing, latency) {
    if (timing) {
      this.detPre.push(timing.pre);
      this.detInfer.push(timing.infer);
      this.detDecode.push(timing.decode);
    }
    this.detLatency.push(latency);
  }

  summary(now) {
    return {
      frameHz: this.frameRate.hz(now),
      tickHz: this.tickRate.hz(now),
      detectHz: this.detectRate.hz(now),
      flowHz: this.flowRate.hz(now),
      tickMs: this.tickMs.value,
      flowMs: this.flowMs.value,
      drawMs: this.drawMs.value,
      inferMs: this.detInfer.value,
      latencyMs: this.detLatency.value,
    };
  }
}

export class DebugPanel {
  constructor(el) {
    this.el = el;
    this.visible = false;
    this.lastRender = 0;
    // 收合狀態：路測時面板佔掉近半個畫面，使用者要能一鍵縮成一條標題列
    //（保留標題列而不是整個關掉 —— 「除錯模式開著」這件事仍然看得見，
    //  且隨時可以點開，不必回工具列找按鈕）。
    this.collapsed = false;
    if (this.el) {
      this.bar = document.createElement('div');
      this.bar.className = 'dp-bar';
      this.body = document.createElement('pre');
      this.body.className = 'dp-body';
      this.el.append(this.bar, this.body);
      this.bar.addEventListener('click', () => this.setCollapsed(!this.collapsed));
      this._renderBar();
    }
  }

  setVisible(v) {
    this.visible = v;
    if (this.el) this.el.style.display = v ? 'block' : 'none';
  }

  setCollapsed(v) {
    this.collapsed = v;
    if (!this.el) return;
    this.el.classList.toggle('collapsed', v);
    this._renderBar();
  }

  _renderBar() {
    if (this.bar) this.bar.textContent = `🔍 除錯資訊 ${this.collapsed ? '▸ 點擊展開' : '▾'}`;
  }

  render(now, metrics, pipeline, detector, extra = []) {
    if (!this.visible || !this.el) return;
    if (this.collapsed) return;                   // 收合時不必組字串
    if (now - this.lastRender < 200) return;      // 面板本身不必每幀重繪
    this.lastRender = now;

    const s = metrics.summary(now);
    const lines = [];
    lines.push(
      `frame ${s.frameHz.toFixed(1)}Hz | tick ${s.tickHz.toFixed(1)}Hz`
      + ` | flow ${s.flowHz.toFixed(1)}Hz | det ${s.detectHz.toFixed(1)}Hz`
    );
    lines.push(
      `tick ${metrics.tickMs.str()}ms (flow ${metrics.flowMs.str()} draw ${metrics.drawMs.str()})`
    );
    lines.push(
      `yolo pre ${metrics.detPre.str()} infer ${metrics.detInfer.str()}`
      + ` dec ${metrics.detDecode.str()} → 延遲 ${metrics.detLatency.str(0)}ms`
    );
    if (detector && detector.info) {
      lines.push(`model ${detector.info.model} @${detector.info.provider} ${detector.info.inputSize}px`);
    }
    lines.push('─'.repeat(34));
    for (const l of pipeline.debugLines()) lines.push(l);
    const st = pipeline.stats;
    const fails = Object.entries(st.flowFail || {})
      .sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k, v]) => `${k}:${v}`).join(' ');
    lines.push(`flow ok ${st.flowOk}/${st.flowOk + Object.values(st.flowFail || {}).reduce((a, b) => a + b, 0)}`
      + (fails ? ` | ${fails}` : ''));
    for (const l of extra) lines.push(l);

    this.body.textContent = lines.join('\n');
  }
}
