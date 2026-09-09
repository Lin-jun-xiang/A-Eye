// =============================================
// 影片分析面板（只在影片模式出現）
// =============================================
// 為什麼需要一個獨立的面板，而不是沿用手機上的徽章：
//
//   徽章是為「開車中掃一眼」設計的 —— 資訊量刻意壓到最低、疊在畫面上、
//   內容會隨狀態變動。用它來看影片分析剛好每一項都是缺點：
//   資訊不夠、會和 overlay 重疊、而且數值一直跳讓人讀不到。
//
//   分析影片的需求相反：資訊要多、位置要固定、**數值不能閃**。
//   所以這裡的三個設計決定是：
//     1. 行的結構固定（標籤 + 值），只改 textContent，不重建 DOM
//     2. 固定 5Hz 更新，且值沒變就不寫 DOM（DOM 寫入才是閃爍的來源）
//     3. 事件用「時間軸」累積，不是「停留 3 秒的徽章」——
//        影片分析最重要的問題是「有沒有在正確的時間點觸發」，
//        而一閃即逝的徽章正好答不了這個問題

const ROWS = [
  ['time', '影片時間'],
  ['ego', '自車'],
  ['target', '前車'],
  ['track', '追蹤品質'],
  ['brake', '剎車燈'],
  ['chmsl', '第三剎車燈'],
  ['level', '燈位準'],
  ['evidence', '起步證據'],
  ['reason', '未觸發原因'],
  ['flow', '光流'],
  ['detect', '偵測'],
];

export class AnalysisPanel {
  constructor(el) {
    this.el = el;
    this.visible = false;
    this.lastRender = 0;
    this.cells = {};
    this.lastText = {};
    this.events = [];
    this._built = false;
  }

  _build() {
    if (this._built || !this.el) return;
    this.el.innerHTML = '';
    for (const [key, label] of ROWS) {
      const row = document.createElement('div');
      row.className = 'ap-row';
      const k = document.createElement('span');
      k.className = 'ap-key';
      k.textContent = label;
      const v = document.createElement('span');
      v.className = 'ap-val';
      v.textContent = '--';
      row.append(k, v);
      this.el.append(row);
      this.cells[key] = v;
    }
    const hr = document.createElement('div');
    hr.className = 'ap-sep';
    hr.textContent = '事件時間軸';
    this.el.append(hr);
    this.log = document.createElement('div');
    this.log.className = 'ap-log';
    this.log.textContent = '（尚無事件）';
    this.el.append(this.log);
    this._built = true;
  }

  setVisible(v) {
    this.visible = v;
    if (!this.el) return;
    if (v) this._build();
    this.el.style.display = v ? 'block' : 'none';
    if (!v) { this.events.length = 0; if (this.log) this.log.textContent = '（尚無事件）'; }
  }

  /** 值沒變就不寫 DOM —— DOM 寫入才是閃爍的來源 */
  _set(key, text) {
    if (this.lastText[key] === text) return;
    this.lastText[key] = text;
    const c = this.cells[key];
    if (c) c.textContent = text;
  }

  /** 事件用累積的時間軸呈現，而不是一閃即逝的徽章 */
  addEvent(videoTime, kind, text) {
    this.events.unshift({ t: videoTime, kind, text });
    if (this.events.length > 12) this.events.pop();
    if (!this.log) return;
    this.log.textContent = this.events
      .map((e) => `${e.t.toFixed(2)}s  ${e.text}`)
      .join('\n');
  }

  render(now, hud, videoTime, duration, detector) {
    if (!this.visible || !this.el) return;
    if (now - this.lastRender < 200) return;        // 固定 5Hz
    this.lastRender = now;

    this._set('time', `${videoTime.toFixed(2)} / ${duration.toFixed(1)} s`);
    // 影片沒有 GPS / IMU，所以自車狀態是靠影片自己的背景尺度判斷的（visual-scale）。
    // 把 z 值一起顯示 —— 否則「為什麼判成行駛中」完全無法診斷。
    const z = hud.bgExpZ;
    this._set('ego', hud.assumeStill
      ? '強制假設靜止'
      : `${hud.egoLabel}（${hud.egoSource}`
        + (z === null || z === undefined ? '' : ` z=${z.toFixed(1)}`) + '）');

    const st = hud.stats;
    const t = hud.target;
    this._set('target', t
      ? `#${t.id}  ${Math.round(t.box.w)}x${Math.round(t.box.h)}px  ${(t.score * 100) | 0}%`
      : '未鎖定');

    // 目標「新鮮度」= 有多少比例的 tick 目標剛被偵測更新過。
    // 這是整條因果鏈的源頭：偵測抓不到 → track 被淘汰重建 → 證據清空。
    // 實車量測過 21%（夜間近距離的白車），沒有這一行完全看不出問題在偵測。
    if (st && st.targetTicks) {
      const k = hud.trackerStats;
      this._set('track', `目標新鮮 ${(st.targetFresh / st.targetTicks * 100).toFixed(0)}%`
        + `　換手 ${st.targetChanges}（清空證據 ${st.evidenceResets}）`
        + (k ? `　偵測框 ${k.dets}→配對 ${k.matched} 新建 ${k.created}` : ''));
    } else {
      this._set('track', '--');
    }

    // 用持續存在的狀態，不用單一 tick 的回傳 —— 分析頻率低於畫面幀率，
    // 若只在有量測的那一幀才有值，面板會在數值與 -- 之間跳
    // lit 與 on 的差別是整個判定的關鍵：只看過一個位準時，無法知道那是
    // 尾燈還是剎車燈（實車量測同一台車兩者差 5.9 倍）。判「熄」有證據，
    // 判「踩著」不能用預設值充當。
    const LBL = {
      on: '踩著（範圍已解析）',
      lit: '亮著，但分不出尾燈/剎車（尚未看過落差）',
      off: '已鬆開',
      unknown: '不確定',
    };
    this._set('brake', (LBL[hud.brakeState] || '--')
      + (hud.brakePrimed ? '（已預備）' : '')
      + (hud.brakeBlinking ? '（閃爍中，抑制）' : ''));

    // 第三剎車燈：可用時它是權威判據，所以要看得到它到底有沒有被找到
    if (hud.brakeChmslUsable) {
      const r = hud.brakeChmslPeak > 0 ? hud.brakeChmslVal / hud.brakeChmslPeak : 0;
      const p = hud.brakeChmsl && hud.brakeChmsl.pos;
      this._set('chmsl', `${hud.brakeChmslState === 'on' ? '亮' : hud.brakeChmslState === 'off' ? '熄' : '?'}`
        + `　${hud.brakeChmslVal.toFixed(0)} / 峰值 ${hud.brakeChmslPeak.toFixed(0)} = ${r.toFixed(2)}`
        + (p ? `　位置 ${p.x.toFixed(2)},${p.y.toFixed(2)}` : '')
        + (hud.brakePressed ? '　已踩下' : ''));
    } else {
      // 三種「不可用」要分開講，否則實測時無法判斷是幾何沒對準還是還沒看到落差
      this._set('chmsl', hud.brakeChmslFar ? '車太遠，暫不使用'
        : !hud.brakeChmslFound ? '找不到（退回左右尾燈判定）'
        : `已找到但還沒看過落差（val ${hud.brakeChmslVal.toFixed(0)}）—— 尚不採用`);
    }

    // 位準與峰值並排 —— 判定「熄滅」靠的就是「位準 / 峰值」這個比值
    if (hud.brakePeak > 0) {
      const r = hud.brakeLevel / hud.brakePeak;
      this._set('level', `${hud.brakeLevel.toFixed(0)} / 峰值 ${hud.brakePeak.toFixed(0)}`
        + ` = ${r.toFixed(2)}　熄滅門檻 ${hud.brakeOffRatio}`);
    } else {
      this._set('level', '尚未建立峰值（需先看到一對對稱紅燈）');
    }

    const d = hud.departure;
    if (d) {
      const z = Math.max(0, Math.min(1, d.z / Math.max(d.zFire, 1e-6)));
      const l = Math.max(0, Math.min(1, d.llr / Math.max(d.sprtA, 1e-6)));
      const ttc = isFinite(d.ttc) && d.ttc < 999 ? `  TTC ${d.ttc.toFixed(1)}s` : '';
      this._set('evidence', `z ${bar(z)} ${(z * 100) | 0}%   LLR ${bar(l)} ${(l * 100) | 0}%${ttc}`);
      this._set('reason', d.reason || '--');
    }

    if (st) {
      const fails = Object.entries(st.flowFail || {}).sort((a, b2) => b2[1] - a[1]);
      const total = st.flowOk + fails.reduce((a, [, v]) => a + v, 0);
      const top = fails.slice(0, 2).map(([k, v]) => `${k}:${v}`).join(' ');
      this._set('flow', `ok ${st.flowOk}/${total}${top ? '  ' + top : ''}`);
      this._set('detect', `${st.detections} 次`
        + (detector && detector.info ? `  ${detector.info.inputSize}px @${detector.info.provider}` : ''));
    }
  }
}

/** 8 格的文字進度條 —— 寬度固定，所以不會因為數值變動而讓整行重排 */
function bar(frac) {
  const n = Math.round(Math.max(0, Math.min(1, frac)) * 8);
  return '█'.repeat(n) + '·'.repeat(8 - n);
}
