// 一次跑完所有離線自我測試： node tools/selftest/all.mjs
import { spawnSync } from 'child_process';
import { readdirSync } from 'fs';

const files = readdirSync('tools/selftest').filter((f) => f.endsWith('.test.mjs')).sort();
let fail = 0;
for (const f of files) {
  console.log('\n' + '='.repeat(62));
  console.log('  ' + f);
  console.log('='.repeat(62));
  const r = spawnSync(process.execPath, ['tools/selftest/' + f], { stdio: 'inherit' });
  if (r.status !== 0) { fail++; console.log(`  ⚠ ${f} 以 exit code ${r.status} 結束`); }
}
console.log('\n' + '='.repeat(62));
console.log(fail === 0 ? `✅ ${files.length} 個測試檔全部執行完成` : `❌ ${fail}/${files.length} 個測試檔失敗`);
process.exit(fail ? 1 : 0);
