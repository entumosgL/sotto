// Одно SSE-подключение на адрес для всех вкладок панели.
// Браузер держит не больше 6 соединений к одному сайту; если каждая вкладка открывает своё вечное
// подключение к событиям, на третьей-четвёртой вкладке страницы начинают «висеть» в очереди.
const streams = new Map(); // url -> {es, ports: Set<MessagePort>}

function subscribe(port, url) {
  let s = streams.get(url);
  if (!s) {
    s = {es: new EventSource(url), ports: new Set()};
    s.es.onmessage = (m) => { for (const p of s.ports) p.postMessage({url, data: m.data}); };
    streams.set(url, s);
  }
  s.ports.add(port);
}

function unsubscribe(port, url) {
  const s = streams.get(url);
  if (!s) return;
  s.ports.delete(port);
  if (!s.ports.size) { s.es.close(); streams.delete(url); }
}

onconnect = (e) => {
  const port = e.ports[0];
  port.onmessage = (m) => (m.data.op === "sub" ? subscribe : unsubscribe)(port, m.data.url);
  port.start();
};
