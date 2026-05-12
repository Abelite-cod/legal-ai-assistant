// Legal AI Assistant — Service Worker
// Provides offline shell + caches static assets for app-like experience

const CACHE_NAME = "legal-ai-v1";
const STATIC_ASSETS = ["/", "/static/manifest.json"];

// Install: cache the app shell
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: network-first for API calls, cache-first for static assets
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Always go to network for API endpoints
  if (url.pathname.startsWith("/ask") ||
      url.pathname.startsWith("/upload_pdf") ||
      url.pathname.startsWith("/sessions") ||
      url.pathname.startsWith("/health")) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Cache-first for everything else (HTML, static files)
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match("/"));
    })
  );
});
