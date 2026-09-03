import { CONFIG, derivedThresholds } from '../../src/config.js';
const t = derivedThresholds(CONFIG);
console.log('zFire =', t.zFire.toFixed(4), '(expect 3.7190 for alpha=1e-4)');
console.log('sprtA =', t.sprtA.toFixed(4), 'sprtB =', t.sprtB.toFixed(4));
const mu = CONFIG.departure.effectSize, zc = CONFIG.departure.zClamp;
const maxPerTick = mu * zc - mu * mu / 2;
console.log('max LLR per tick =', maxPerTick.toFixed(3), '→ min ticks to fire =', Math.ceil(t.sprtA / maxPerTick), '(config minTicks =', CONFIG.departure.minTicks + ')');
