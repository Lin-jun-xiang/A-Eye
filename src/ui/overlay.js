// =============================================
// Overlay 繪製
// =============================================
// 只負責畫，不做任何判定。除了框以外還畫出幾個「看得見的內部狀態」：
// 地平線、走廊、光流 ROI、以及起步證據的進度條 ——
// 在手機上實測時，看得見證據怎麼累積才有可能除錯。

const CLASS_NAME = { 2: 'car', 5: 'bus', 7: 'truck', 9: 'light' };

export class Overlay {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.video = video;
    // 必須與 CSS 的 object-fit 一致，否則框會畫錯位置。
    // 相機模式用 cover（全螢幕取景），影片模式用 contain（要看到完整畫面）。
    this.fit = 'cover';
  }

  /** 把 canvas 對齊到 video 的實際顯示區域 */
  syncSize(vw, vh) {
    const rect = this.video.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(rect.width), h = Math.round(rect.height);
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';

    // cover = 等比縮放後裁切（取 max）；contain = 完整塞進去、留黑邊（取 min）
    const scale = this.fit === 'contain'
      ? Math.min(w / vw, h / vh)
      : Math.max(w / vw, h / vh);
    this.map = {
      scale: scale * dpr,
      dx: (w - vw * scale) / 2 * dpr,
      dy: (h - vh * scale) / 2 * dpr,
      w: w * dpr, h: h * dpr,
    };
  }

  _pt(x, y) {
    return [this.map.dx + x * this.map.scale, this.map.dy + y * this.map.scale];
  }

  clear() { this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }

  draw(hud, vw, vh, { debug = false } = {}) {
    this.syncSize(vw, vh);
    const ctx = this.ctx;
    this.clear();

    if (debug) {
      this._drawCorridor(hud, vw, vh);
      this._drawHorizon(hud, vw, vh);
      this._drawEgoRegions(hud, vw, vh);
    }

    // 所有 track
    for (const t of hud.tracks || []) {
      const isTarget = hud.target && t.id === hud.target.id;
      if (!isTarget && !debug) continue;
      this._drawBox(t.box, {
        color: isTarget ? '#4ade80' : (t.classId === 9 ? '#fbbf24' : 'rgba(255,255,255,0.35)'),
        width: isTarget ? 3 : 1.5,
        dash: t.coasting ? [6, 4] : null,
        label: debug
          ? `#${t.id} ${CLASS_NAME[t.classId] || t.classId} ${(t.score * 100) | 0}%`
          : null,
      });
    }

    // 紅綠燈框
    if (hud.light && hud.light.box) {
      const c = hud.lightColor === 'green' ? '#22c55e'
        : hud.lightColor === 'red' ? '#ef4444'
        : hud.lightColor === 'yellow' ? '#eab308' : 'rgba(255,255,255,0.4)';
      this._drawBox(hud.light.box, { color: c, width: 2.5 });
    }

    // 光流 ROI（debug）
    if (debug && hud.lastFlow && hud.lastFlow.roi) {
      const r = hud.lastFlow.roi;
      this._drawBox({ x: r.x, y: r.y, w: r.w, h: r.h }, {
        color: hud.trusted ? 'rgba(56,189,248,0.8)' : 'rgba(239,68,68,0.8)',
        width: 1.5, dash: [4, 4],
        label: `ROI ${r.w}x${r.h} @${(r.scale * 100).toFixed(0)}%`,
      });
    }

    // 剎車燈取樣區（debug）
    if (debug && hud.target) this._drawLampRois(hud);

    // 起步證據進度條
    if (hud.target && hud.departure) this._drawEvidence(hud);
  }

  _drawBox(box, { color, width, dash, label }) {
    const ctx = this.ctx;
    const [x, y] = this._pt(box.x, box.y);
    const w = box.w * this.map.scale, h = box.h * this.map.scale;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.strokeRect(x, y, w, h);
    if (label) {
      ctx.setLineDash([]);
      ctx.font = '600 11px system-ui, sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(x, y - 15, tw + 8, 15);
      ctx.fillStyle = color;
      ctx.fillText(label, x + 4, y - 4);
    }
    ctx.restore();
  }

  /** 已學到的「自車結構」區域（引擎蓋 / 儀表板 / 反光）—— 這些框永不選為前車 */
  _drawEgoRegions(hud, vw, vh) {
    for (const r of hud.egoRegions || []) {
      this._drawBox({ x: r.x * vw, y: r.y * vh, w: r.w * vw, h: r.h * vh }, {
        color: 'rgba(148,163,184,0.7)', width: 1.5, dash: [3, 3], label: '自車結構',
      });
    }
  }

  _drawHorizon(hud, vw, vh) {
    const ctx = this.ctx;
    const [x0, y0] = this._pt(0, hud.horizon * vh);
    const [x1] = this._pt(vw, 0);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText('horizon', x0 + 6, y0 - 4);
    ctx.restore();
  }

  _drawCorridor(hud, vw, vh) {
    if (!hud.corridor || hud.corridor.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    hud.corridor.forEach((p, i) => {
      const [x, y] = this._pt(p.left * vw, p.yr * vh);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    for (let i = hud.corridor.length - 1; i >= 0; i--) {
      const p = hud.corridor[i];
      const [x, y] = this._pt(p.right * vw, p.yr * vh);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(74,222,128,0.07)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(74,222,128,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  /** 剎車燈的兩個取樣區與中央車身參考區 —— 看得見才能判斷 ROI 有沒有對準 */
  _drawLampRois(hud) {
    const b = hud.brakeCfg;
    if (!b) return;
    const box = hud.target.box;
    const y0 = box.y + box.h * b.yTop;
    const hh = box.h * (b.yBottom - b.yTop);
    const sw = box.w * b.sideFrac;
    const cw = box.w * b.centerFrac;
    const on = hud.brakeState === 'on';
    const col = on ? 'rgba(239,68,68,0.9)' : 'rgba(148,163,184,0.7)';
    this._drawBox({ x: box.x, y: y0, w: sw, h: hh }, { color: col, width: 1 });
    this._drawBox({ x: box.x + box.w - sw, y: y0, w: sw, h: hh }, { color: col, width: 1 });
    this._drawBox({ x: box.x + (box.w - cw) / 2, y: y0, w: cw, h: hh },
      { color: 'rgba(255,255,255,0.3)', width: 1, dash: [2, 3] });
  }

  /** 把「證據累積到哪了」畫成兩條進度條：KF 的 z 與 SPRT 的 LLR */
  _drawEvidence(hud) {
    const d = hud.departure;
    const ctx = this.ctx;
    const box = hud.target.box;
    const [x, y] = this._pt(box.x, box.y + box.h);
    const w = Math.max(box.w * this.map.scale, 90);

    const bars = [
      { label: 'z', v: d.z / Math.max(d.zFire, 1e-6), color: '#38bdf8' },
      { label: 'LLR', v: d.llr / Math.max(d.sprtA, 1e-6), color: '#a78bfa' },
    ];
    ctx.save();
    ctx.font = '9px system-ui, sans-serif';
    bars.forEach((b, i) => {
      const by = y + 6 + i * 11;
      const frac = Math.max(0, Math.min(1, b.v));
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(x, by, w, 7);
      ctx.fillStyle = frac >= 1 ? '#4ade80' : b.color;
      ctx.fillRect(x, by, w * frac, 7);
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(b.label, x + w + 4, by + 7);
    });
    const ttc = isFinite(d.ttc) && d.ttc < 999 ? `TTC ${d.ttc.toFixed(1)}s` : '';
    if (ttc) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText(ttc, x, y + 6 + bars.length * 11 + 8);
    }
    ctx.restore();
  }
}
