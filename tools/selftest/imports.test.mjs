// 檢查所有 import 路徑都真的存在，且沒有殘留的舊模組引用
import { readFileSync, existsSync, readdirSync, statSync } from 'fs';
import { dirname, resolve, relative } from 'path';

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
// 檢查 HTML 引用
for (const h of ['index.html', 'replay.html']) {
  const src = readFileSync(h, 'utf8');
  const re = /src="(\.?\/?src\/[^"]+)"/g;
  let m;
  while ((m = re.exec(src))) {
    const p = m[1].replace(/^\.\//, '');
    console.log(existsSync(p) ? `✅ ${h} → ${m[1]}` : `❌ ${h} → ${m[1]} (不存在)`);
  }
}
process.exit(bad ? 1 : 0);
