// 檢查所有 import 路徑都真的存在，且沒有殘留的舊模組引用
import { readFileSync, existsSync, readdirSync, statSync } from 'fs';
import { dirname, resolve, relative, sep } from 'path';

function walk(d, out = []) {
  for (const e of readdirSync(d)) {
    const p = d + '/' + e;
    statSync(p).isDirectory() ? walk(p, out) : (e.endsWith('.js') && out.push(p));
  }
  return out;
}
const root = resolve('.');
const files = walk('src');
let bad = 0, n = 0;
for (const f of files) {
  const src = readFileSync(f, 'utf8');
  const re = /(?:from|import)\s+['"](\.[^'"]+)['"]/g;
  let m;
  while ((m = re.exec(src))) {
    n++;
    const target = resolve(dirname(f), m[1]);
    if (!existsSync(target)) { console.log(`❌ ${relative(root, f)} → ${m[1]}  (不存在)`); bad++; }
  }
  // new URL('./x.js', import.meta.url) 形式
  const re2 = /new URL\(\s*['"](\.[^'"]+)['"]/g;
  while ((m = re2.exec(src))) {
    n++;
    const target = resolve(dirname(f), m[1]);
    if (!existsSync(target) && !m[1].endsWith('/')) { console.log(`❌ ${relative(root, f)} → new URL ${m[1]}  (不存在)`); bad++; }
  }
}
console.log(`檢查 ${files.length} 個模組、${n} 個相對引用 → ${bad} 個錯誤`);
// 檢查 HTML 的引用（含 <script src> 與 ESM 的 import '...'）
for (const h of ['index.html', 'replay.html', 'analyze.html']) {
  if (!existsSync(h)) { console.log(`❌ ${h} 不存在`); bad++; continue; }
  const src = readFileSync(h, 'utf8');
  const seen = new Set();
  for (const re of [/src="(\.?\/?src\/[^"]+)"/g, /from\s+['"](\.\/src\/[^'"]+)['"]/g]) {
    let m;
    while ((m = re.exec(src))) {
      const p = m[1].replace(/^\.\//, '');
      if (seen.has(p)) continue;
      seen.add(p);
      if (existsSync(p)) console.log(`✅ ${h} → ${m[1]}`);
      else { console.log(`❌ ${h} → ${m[1]} (不存在)`); bad++; }
    }
  }
  if (!seen.size) { console.log(`❌ ${h} 沒有偵測到任何模組引用`); bad++; }
}

// Service Worker 的預快取清單必須與實際檔案一致。
// 它是 network-first，所以少一個檔案線上不會壞 —— 但離線會壞，
// 而且是「某一個模組 404 → 整個 App 起不來」這種難查的壞法。
// 反向也要查：src/ 下的模組若沒進清單，離線就少一塊。
{
  const sw = readFileSync('sw.js', 'utf8');
  const listed = new Set();
  const re = /'\.\/([^']+)'/g;
  let m;
  while ((m = re.exec(sw.slice(sw.indexOf('const FILES'), sw.indexOf('];', sw.indexOf('const FILES')))))) {
    listed.add(m[1]);
    if (!existsSync(m[1])) { console.log(`❌ sw.js 清單裡的 ${m[1]} 不存在`); bad++; }
  }
  for (const f of files) {
    const rel = relative(root, f).split(sep).join('/');
    if (!listed.has(rel)) { console.log(`❌ sw.js 預快取清單漏了 ${rel}`); bad++; }
  }
  console.log(`sw.js 預快取 ${listed.size} 個檔案 → ${bad === 0 ? '與實際檔案一致' : '有問題'}`);
}
process.exit(bad ? 1 : 0);
