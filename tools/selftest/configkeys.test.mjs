// 檢查程式碼裡讀到的 config 路徑都真的存在
// （config 讀到 undefined 會靜默變成 NaN，是最難查的一類 bug）
// 只匹配明確的 config 前綴：CONFIG.x.y / cfg.x.y / this.cfg.x.y

import { CONFIG } from '../../src/config.js';
import { readFileSync, readdirSync, statSync } from 'fs';

function walk(d, o = []) {
  for (const e of readdirSync(d)) {
    const p = d + '/' + e;
    statSync(p).isDirectory() ? walk(p, o) : (e.endsWith('.js') && o.push(p));
  }
  return o;
}

// 抓 (this.cfg|cfg|CONFIG).<group>.<key>
const RE = /(?:this\.cfg|\bcfg|\bCONFIG)\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)/g;
// 抓解構出中間變數的形式：const t = cfg.tracker;  → 之後的 t.xxx 不檢查（保守略過）

let bad = 0, checked = 0;
const seen = new Set();
for (const f of walk('src')) {
  if (f.endsWith('config.js')) continue;
  const src = readFileSync(f, 'utf8');
  let m;
  while ((m = RE.exec(src))) {
    const [, group, key] = m;
    checked++;
    seen.add(`${group}.${key}`);
    if (!(group in CONFIG)) { console.log(`❌ ${f}: CONFIG.${group} 群組不存在`); bad++; continue; }
    const g = CONFIG[group];
    if (typeof g !== 'object' || Array.isArray(g)) continue;   // 純值群組，第二段是方法呼叫
    if (!(key in g)) { console.log(`❌ ${f}: CONFIG.${group}.${key} 不存在`); bad++; }
  }
}
console.log(`  檢查 ${checked} 個明確的 config 讀取（${seen.size} 個不同路徑）→ ${bad} 個錯誤`);

// 反向檢查：config 裡定義了但整個 src/ 都沒讀到的鍵（可能是死參數）
// 注意要排除 config.js 自己，否則每個鍵都會自我匹配
const allSrc = walk('src')
  .filter((f) => !f.endsWith('config.js'))
  .map((f) => readFileSync(f, 'utf8'))
  .join('\n');
const unused = [];
// 遞迴：巢狀群組（frontCar.plausibility.*、frontCar.egoStructure.*）也要檢查，
// 否則新加的巢狀死參數根本不會被發現
const scan = (obj, path) => {
  for (const [key, v] of Object.entries(obj)) {
    if (v && typeof v === 'object' && !Array.isArray(v)) { scan(v, `${path}.${key}`); continue; }
    if (!new RegExp('\\b' + key + '\\b').test(allSrc)) unused.push(`${path}.${key}`);
  }
};
for (const [group, g] of Object.entries(CONFIG)) {
  if (typeof g !== 'object' || Array.isArray(g)) continue;
  scan(g, group);
}
console.log(unused.length
  ? `  ⚠ 定義了但沒被讀取的參數：${unused.join(', ')}`
  : '  ✅ 沒有未被使用的參數');
process.exit(bad ? 1 : 0);
