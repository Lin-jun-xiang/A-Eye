import * as m from '../../src/util/math.js';
console.log('normInv(1e-4) =', m.normInv(1e-4).toFixed(4), '(expect ~-3.719)');
const k = new m.Kf1d({ q: 1e-4 });
k.update(0);
for (let i = 0; i < 40; i++) { k.predict(0.05); k.update(-0.01 * (i + 1), 1e-6); }
console.log('KF vel =', k.v.toFixed(4), '(expect ~-0.20)');
const r = new m.Rls(3);
for (let i = 0; i < 300; i++) { const u = [Math.random(), Math.random(), Math.random()]; r.update(u, 3*u[0] - 2*u[1] + 0.5*u[2]); }
console.log('RLS w =', Array.from(r.w).map(x => x.toFixed(3)).join(', '), '(expect 3, -2, 0.5) quality =', r.quality.toFixed(3));
console.log('madSigma of N(0,2) sample =', m.madSigma(Array.from({length: 4000}, () => 2 * (Math.sqrt(-2*Math.log(Math.random()))*Math.cos(2*Math.PI*Math.random())))).toFixed(3), '(expect ~2.0)');
console.log('iou =', m.iou({x:0,y:0,w:10,h:10},{x:5,y:0,w:10,h:10}).toFixed(4), '(expect 0.3333)');
