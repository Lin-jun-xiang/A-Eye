#!/usr/bin/env python3
# =============================================
# A-Eye 品牌記號「Lock Bracket」— 圖檔產生器
# =============================================
# mark 只有一份幾何定義（下面的 48x48 座標系），
# 由這支腳本同時輸出：
#   icon.svg        向量版，給 favicon 與 index.html 的 inline sprite
#   icon-192.png    PWA / apple-touch-icon
#   icon-512.png    PWA maskable
#
# 為什麼不用 SVG 轉檔工具：mark 只有四段圓角折線 + 一個圓，
# 用 PIL 直接畫（8x 超採樣後縮小）比引入 cairo 依賴乾淨，
# 而且輸出結果與 SVG 版本逐像素一致。
#
#   python tools/make_icons.py

import math
from PIL import Image, ImageDraw

# ---------- mark 幾何（48x48 座標系，與 icon.svg / index.html sprite 同源） ----------
STROKE = 4.0          # 描邊寬
CORNER_R = 4.0        # 四角圓弧半徑
EDGE = 7.0            # 框邊距畫布邊緣
ARM = 10.0            # 每支角的直線臂長（EDGE → EDGE+ARM）
PUPIL_R = 7.0         # 中央實心瞳孔半徑

BG_TOP = (0x1b, 0x3a, 0x6b)      # 品牌深藍（沿用舊 icon 的底色）
BG_BOTTOM = (0x12, 0x29, 0x4c)
FG = (0xff, 0xff, 0xff)


def _corner_path(flip_x: bool, flip_y: bool, steps: int = 14):
    """單支角的折線點列：直線 → 圓弧 → 直線。回傳 48 座標系的點。"""
    e, a, r = EDGE, ARM, CORNER_R
    pts = [(e + a, e)]                        # 水平臂外端
    cx, cy = e + r, e + r                     # 圓弧圓心
    for i in range(steps + 1):                # 270° → 180°（左上象限）
        th = math.radians(270 - 90 * i / steps)
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    pts.append((e, e + a))                    # 垂直臂外端

    if flip_x:
        pts = [(48 - x, y) for x, y in pts]
    if flip_y:
        pts = [(x, 48 - y) for x, y in pts]
    return pts


CORNERS = [_corner_path(fx, fy) for fx, fy in
           ((False, False), (True, False), (True, True), (False, True))]


# ---------- PNG ----------
def render_png(size: int, path: str, ss: int = 8, mark_ratio: float = 0.72):
    """size: 輸出邊長。mark_ratio: mark 的 48 單位框佔畫布比例。

    mark 實際著墨範圍是 48 框的 79%（EDGE-STROKE/2 到對邊），
    所以 0.72 的框 → 實際佔畫布約 57%，穩穩落在 maskable 的內側 80% 安全區。
    """
    S = size * ss
    img = Image.new('RGB', (S, S))

    # 垂直漸層底
    d = ImageDraw.Draw(img)
    for y in range(S):
        t = y / max(S - 1, 1)
        d.line([(0, y), (S, y)],
               fill=tuple(round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)))

    # 48 座標系 → 像素
    box = S * mark_ratio
    off = (S - box) / 2
    u = box / 48.0
    def P(p):
        return (off + p[0] * u, off + p[1] * u)

    w = STROKE * u
    for pts in CORNERS:
        px = [P(p) for p in pts]
        d.line(px, fill=FG, width=round(w), joint='curve')
        # PIL 沒有 round cap，兩端各補一顆圓點
        for cap in (px[0], px[-1]):
            d.ellipse([cap[0] - w / 2, cap[1] - w / 2, cap[0] + w / 2, cap[1] + w / 2], fill=FG)

    c = P((24, 24))
    pr = PUPIL_R * u
    d.ellipse([c[0] - pr, c[1] - pr, c[0] + pr, c[1] + pr], fill=FG)

    img.resize((size, size), Image.LANCZOS).save(path, optimize=True)
    print(f'  {path}  {size}x{size}')


# ---------- SVG ----------
SVG_MARK = (
    '<g fill="none" stroke="currentColor" stroke-width="4" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M17 7H11a4 4 0 0 0-4 4v6"/>'
    '<path d="M31 7h6a4 4 0 0 1 4 4v6"/>'
    '<path d="M41 31v6a4 4 0 0 1-4 4h-6"/>'
    '<path d="M7 31v6a4 4 0 0 0 4 4h6"/>'
    '</g><circle cx="24" cy="24" r="7" fill="currentColor"/>'
)


def render_svg(path: str):
    # favicon 用：帶品牌底色的完整圖，mark 縮到 72% 置中（與 PNG 同比例）
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#{BG_TOP[0]:02x}{BG_TOP[1]:02x}{BG_TOP[2]:02x}"/>'
        f'<stop offset="1" stop-color="#{BG_BOTTOM[0]:02x}{BG_BOTTOM[1]:02x}{BG_BOTTOM[2]:02x}"/>'
        '</stop></linearGradient></defs>'
        '<rect width="48" height="48" fill="url(#g)"/>'
        '<g transform="translate(6.72 6.72) scale(0.72)" color="#fff">'
        + SVG_MARK +
        '</g></svg>'
    ).replace('</stop></linearGradient>', '</linearGradient>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f'  {path}')


if __name__ == '__main__':
    print('A-Eye Lock Bracket → 圖檔輸出：')
    render_svg('icon.svg')
    render_png(192, 'icon-192.png')
    render_png(512, 'icon-512.png')
    print('完成。index.html 的 inline sprite 請與 SVG_MARK 保持一致。')
