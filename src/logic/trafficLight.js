// =============================================
// 紅綠燈判定
// =============================================
// 相對 v6 的改動：
//   * 鎖定改用 tracker 的 KF track（有慣性滑行，短暫遮擋不會斷）
//   * 色彩判定加入「燈位佐證」：垂直式號誌紅燈在上、綠燈在下。
//     純色彩門檻很容易把黃燈、落日、紅色招牌誤判成紅燈；
//     加上亮點的相對位置後，誤判大幅減少。
//   * 確認條件從「連續 N 幀」改成「持續 N 毫秒」——幀率會變，時間不會。

import { CONFIG } from '../config.js';

export class TrafficLightDetector {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.canvas = typeof document !== 'undefined'
      ? document.createElement('canvas')
      : new OffscreenCanvas(32, 32);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.reset();
  }

  reset() {
    this.lockedId = null;
    this.state = 'unknown';       // 最後確認的顏色
    this.pendingGreenSince = 0;
    this.lastFireTs = -Infinity;
    this.lastColor = 'unknown';
    this.lastDetail = '';
  }

  /**
   * 分析 bbox 內的燈色。
   * @returns 'red' | 'green' | 'yellow' | 'unknown'
   */
  analyze(source, box) {
    const w = Math.max(1, Math.min(96, Math.round(box.w)));
    const h = Math.max(1, Math.min(96, Math.round(box.h)));
    if (box.w < 4 || box.h < 4) return 'unknown';
    this.canvas.width = w;
    this.canvas.height = h;
    try {
      this.ctx.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, w, h);
    } catch (e) { return 'unknown'; }
    const d = this.ctx.getImageData(0, 0, w, h).data;

    // 垂直式（高>寬）用 y 分段；橫式用 x 分段
    const vertical = h >= w;
    let redN = 0, greenN = 0, yellowN = 0, bright = 0;
    let redPos = 0, greenPos = 0;

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const v = mx / 255;
        const sat = mx > 0 ? (mx - mn) / mx : 0;
        // 只看「亮且有飽和度」的像素——號誌燈是自發光的
        if (v < 0.45 || sat < 0.25) continue;
        bright++;
        const pos = vertical ? py / h : px / w;
        if (r > 150 && r > g * 1.45 && r > b * 1.4) { redN++; redPos += pos; }
        else if (g > 100 && g > r * 1.15 && g > b * 1.05) { greenN++; greenPos += pos; }
        else if (r > 140 && g > 110 && b < Math.min(r, g) * 0.6) yellowN++;
      }
    }

    if (bright < this.cfg.light.minBrightPixels) { this.lastDetail = `bright=${bright}`; return 'unknown'; }
    redPos = redN ? redPos / redN : -1;
    greenPos = greenN ? greenPos / greenN : -1;

    // 燈位佐證：紅在上（pos 小）、綠在下（pos 大）
    // 位置不符時把票數打折，而不是直接否決（號誌型式很多）
    const redScore = redN * (redPos >= 0 && redPos < 0.55 ? 1 : 0.45);
    const greenScore = greenN * (greenPos >= 0 && greenPos > 0.45 ? 1 : 0.45);
    const yellowScore = yellowN * 0.8;

    this.lastDetail = `b=${bright} r=${redN}@${redPos.toFixed(2)} g=${greenN}@${greenPos.toFixed(2)} y=${yellowN}`;

    const minFrac = bright * 0.12;
    if (greenScore > minFrac && greenScore > redScore * 1.6 && greenScore > yellowScore) return 'green';
    if (redScore > minFrac && redScore > greenScore * 1.6 && redScore > yellowScore) return 'red';
    if (yellowScore > minFrac && yellowScore > redScore && yellowScore > greenScore) return 'yellow';
    return 'unknown';
  }

  /**
   * @param lightTracks  class=9 的已確認 track
   * @returns { fired, color, box }
   */
  update(source, lightTracks, vw, vh, now, canAlert) {
    const c = this.cfg.light;

    // ---- 鎖定：優先沿用上次鎖定的 track ----
    let target = null;
    if (this.lockedId !== null) target = lightTracks.find((t) => t.id === this.lockedId) || null;
    if (!target) {
      // 選面積大且靠畫面中央的（號誌在正前方上空）
      let best = null, bestScore = 0;
      for (const t of lightTracks) {
        const b = t.boxAt(now);
        const area = (b.w * b.h) / (vw * vh);
        const cx = (b.x + b.w / 2) / vw;
        const dev = (cx - 0.5) / 0.3;
        const s = area * Math.exp(-0.5 * dev * dev);
        if (s > bestScore) { bestScore = s; best = t; }
      }
      target = best;
      this.lockedId = best ? best.id : null;
    }

    if (!target) {
      this.pendingGreenSince = 0;
      this.lastColor = 'unknown';
      return { fired: false, color: 'unknown', box: null };
    }

    const box = target.boxAt(now);
    const color = this.analyze(source, box);
    this.lastColor = color;

    let fired = false;
    // 紅 → 綠 的轉換，且需持續 confirmMs
    if (color === 'green' && this.state === 'red') {
      if (!this.pendingGreenSince) this.pendingGreenSince = now;
      if (now - this.pendingGreenSince >= c.confirmMs) {
        if (canAlert && now - this.lastFireTs > c.cooldownMs) {
          fired = true;
          this.lastFireTs = now;
        }
        this.state = 'green';
        this.pendingGreenSince = 0;
      }
    } else {
      this.pendingGreenSince = 0;
      if (color !== 'unknown') this.state = color;
    }

    return { fired, color, box };
  }

  debugLine() {
    return `light=${this.lastColor}(state=${this.state}) ${this.lastDetail}`;
  }
}
