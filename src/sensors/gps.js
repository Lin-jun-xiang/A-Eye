// =============================================
// GPS 測速
// =============================================
// 注意：pos.coords.speed 在靜止時大量瀏覽器回傳 null，都市高樓間更差。
// 因此本模組明確回報「無資料」，由 egoMotion 融合層決定怎麼處理，
// 絕不讓呼叫端沿用一個過期的 boolean（v6 的做法會導致行駛中一路誤報，
// 或反過來永久靜默）。

import { CONFIG } from '../config.js';

export class GpsSensor {
  constructor(cfg = CONFIG) {
    this.cfg = cfg;
    this.watchId = null;
    this.speed = null;        // m/s
    this.ts = 0;
    this.accuracy = null;
    this.error = null;
    this.onUpdate = null;
  }

  start() {
    if (this.watchId !== null) return;
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      this.error = 'unsupported';
      return;
    }
    this.watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const s = pos.coords.speed;
        this.speed = (s !== null && s !== undefined && s >= 0 && isFinite(s)) ? s : null;
        this.accuracy = pos.coords.accuracy;
        this.ts = performance.now();
        this.error = null;
        if (this.onUpdate) this.onUpdate(this.speed);
      },
      (err) => { this.error = err.message; this.speed = null; },
      { enableHighAccuracy: true, maximumAge: 1500, timeout: 6000 }
    );
  }

  stop() {
    if (this.watchId !== null) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }
    this.speed = null;
    this.ts = 0;
  }

  /** 目前可用的速度（m/s）；無資料或過期回 null */
  currentSpeed(now = performance.now()) {
    if (this.speed === null) return null;
    if (now - this.ts > this.cfg.ego.gpsStaleMs) return null;
    return this.speed;
  }
}
