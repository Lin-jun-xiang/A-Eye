// A-Eye Service Worker
// 策略：network-first + cache fallback。
// 對這個 App 來說 network-first 是刻意的：模組檔案很多，
// stale-while-revalidate 很容易讓不同版本的模組混在一起（極難除錯）。
// 離線時才回退到快取。

const CACHE = 'aeye-v8';

const FILES = [
  './',
  './index.html',
  './replay.html',
  './manifest.json',
  './src/main.js',
  './src/config.js',
  './src/util/math.js',
  './src/core/pipeline.js',
  './src/sensors/gps.js',
  './src/sensors/imu.js',
  './src/perception/detector.js',
  './src/perception/detector.worker.js',
  './src/tracking/kalmanBox.js',
  './src/tracking/tracker.js',
  './src/motion/opticalFlow.js',
  './src/motion/departure.js',
  './src/motion/egoMotion.js',
  './src/logic/frontCar.js',
  './src/logic/trafficLight.js',
  './src/ui/alerts.js',
  './src/ui/overlay.js',
  './src/ui/hud.js',
  './src/capture/frameSource.js',
  './src/capture/recorder.js',
  './src/tools/replay.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      // 單一檔案失敗不該讓整個安裝失敗
      Promise.allSettled(FILES.map((f) => c.add(f)))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  e.respondWith(
    fetch(req)
      .then((r) => {
        // 只快取同源且成功的回應（模型與 CDN 的大檔交給瀏覽器 HTTP cache）
        if (r.ok && new URL(req.url).origin === location.origin) {
          const clone = r.clone();
          caches.open(CACHE).then((c) => c.put(req, clone)).catch(() => {});
        }
        return r;
      })
      .catch(() => caches.match(req))
  );
});
