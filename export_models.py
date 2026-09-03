"""
A-Eye v7 模型匯出腳本
=====================
匯出前車偵測用的 YOLO ONNX 模型到 models/ 目錄。

使用方式：
  pip install ultralytics onnx onnxsim
  python export_models.py

產出（依 src/config.js 的 modelCandidates 順序被載入）：
  models/yolov8n_384.onnx   (~12 MB)  ← v7 主要模型，384px 靜態輸入
  models/yolov8n.onnx       (~13 MB)  ← 640px，退路
  models/yolov8s.onnx       (~45 MB)  ← 最後退路

為什麼 384 而不是 640：
  推論成本大致與輸入面積成正比，(640/384)^2 ≈ 2.8 倍。
  對「10~30m 的前車」這個任務，384px 的解析度綽綽有餘
  —— 10m 外的車在 384px 寬的畫面上還有約 60px 寬。
  省下來的時間直接換成更高的 tick 率，而光流的品質對 tick 率極度敏感
  （LK 的 small-motion 假設在 dt 大時會崩潰）。

MiDaS 深度模型在 v7 已移除（檔案也已從 repo 刪除）：
  「最前方的車」改用透視幾何（bbox 底邊越低 = 越近）判斷，
  比 relative inverse depth 穩定、免費、且不佔用推論時間。
"""

import os
import shutil
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 / total_size if total_size > 0 else 0
    print(f"\r  進度: {pct:.1f}%", end="", flush=True)


def export_yolo(weights="yolov8n.pt", imgsz=384, out_name=None):
    """把 YOLOv8 匯出成固定輸入尺寸的 ONNX。

    刻意使用「靜態」輸入軸：動態軸會讓 ORT 無法做部分圖最佳化，
    在 WASM / WebGPU 上都明顯較慢。src/perception/detector.worker.js
    會讀模型自身的輸入尺寸，所以靜態軸不影響前端相容性。
    """
    out_name = out_name or f"{os.path.splitext(weights)[0]}_{imgsz}.onnx"
    out = os.path.join(MODELS_DIR, out_name)
    if os.path.exists(out):
        print(f"[YOLO] 已存在: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
        return True

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[YOLO] ✗ 未安裝 ultralytics（pip install ultralytics）")
        return False

    print(f"[YOLO] 匯出 {weights} @ {imgsz}px → ONNX ...")
    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, simplify=True,
                        opset=17, dynamic=False)
    src = str(path) if path else f"{os.path.splitext(weights)[0]}.onnx"
    if not os.path.exists(src):
        print(f"[YOLO] ✗ 匯出失敗，找不到 {src}")
        return False
    shutil.move(src, out)
    print(f"[YOLO] ✓ {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return True


def download_fallback():
    """沒有 ultralytics 時，抓一份預先匯出的 yolov8n（640px）當退路。"""
    out = os.path.join(MODELS_DIR, "yolov8n.onnx")
    if os.path.exists(out):
        print(f"[YOLO] 退路模型已存在: {out}")
        return
    url = "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx"
    try:
        print(f"[YOLO] 下載退路模型: {url}")
        urllib.request.urlretrieve(url, out, reporthook=progress_hook)
        print(f"\n[YOLO] ✓ {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"\n[YOLO] ✗ 下載失敗: {e}")


if __name__ == "__main__":
    print("=" * 56)
    print("A-Eye v7 模型匯出工具")
    print("=" * 56)

    ok = export_yolo("yolov8n.pt", 384)
    if not ok:
        download_fallback()

    print()
    print("（可選）也想留一份 640px 版當退路：")
    print("  python -c \"from ultralytics import YOLO;"
          " YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True, opset=17)\"")
    print()

    print("models/ 現況：")
    for f in sorted(os.listdir(MODELS_DIR)):
        fp = os.path.join(MODELS_DIR, f)
        note = "  ← 主要模型" if "384" in f else "  ← 退路"
        print(f"  {f:24s} {os.path.getsize(fp) / 1e6:6.1f} MB{note}")
