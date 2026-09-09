// A-Eye Service Worker
// 策略：network-first + cache fallback。
// 對這個 App 來說 network-first 是刻意的：模組檔案很多，
// stale-while-revalidate 很容易讓不同版本的模組混在一起（極難除錯）。
// 離線時才回退到快取。

const CACHE = 'aeye-v11';

const FILES = [
  './',
  './index.html',
  './replay.html',
  './analyze.html',
  './manifest.json',
  './icon.svg',
  './icon-192.png',
  './icon-512.png',
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
  './src/logic/brakeLight.js',
  './src/ui/alerts.js',
  './src/ui/overlay.js',
  './src/ui/hud.js',
  './src/ui/analysisPanel.js',
  './src/capture/frameSource.js',
  './src/capture/recorder.js',
  './src/tools/replay.js',
  './src/tools/analyze.js',
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

  // 跨來源請求一律不攔截，直接讓瀏覽器自己去抓。
  //
  // 這不是效能考量，是正確性問題：SW 用 fetch() 重發跨來源的 no-cors 請求，
  // 拿回來的是 opaque response。而 HTML 規範明確禁止 importScripts() 接受
  // 由 Service Worker 提供的 opaque response，WebKit 會直接丟出
  //   「Network response is CORS-cross-origin」
  // 於是 worker 裡的 importScripts('https://cdn.jsdelivr.net/...ort.min.js')
  // 一定失敗 —— 即使 CDN 本身送了 access-control-allow-origin: *。
  //
  // （v6 沒踩到是因為它用 <script> 標籤在主執行緒載入 ORT，
  //   而 <script> 是接受 opaque response 的。）
  //
  // 不呼叫 respondWith() 就等於「交還給瀏覽器預設行為」，
  // CDN 的大檔本來也該交給瀏覽器的 HTTP cache 管。
  let sameOrigin;
  try {
    sameOrigin = new URL(req.url).origin === location.origin;
  } catch (_) {
    return;
  }
  if (!sameOrigin) return;

  e.respondWith(
    fetch(req)
      .then((r) => {
        if (r.ok) {
          const clone = r.clone();
          caches.open(CACHE).then((c) => c.put(req, clone)).catch(() => {});
        }
        return r;
      })
      // 離線時回退到快取；連快取都沒有就明確回一個 503，
      // 不要回 undefined（那會變成難以診斷的 network error）
      .catch(async () => {
        const hit = await caches.match(req);
        return hit || new Response('offline and not cached', {
          status: 503,
          headers: { 'content-type': 'text/plain; charset=utf-8' },
        });
      })
  );
});
