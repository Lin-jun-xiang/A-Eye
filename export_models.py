"""
A-Eye 模型匯出腳本
==================
匯出 YOLOv8n 和 MiDaS Small 的 ONNX 模型到 models/ 目錄。

使用方式：
  pip install ultralytics onnx onnxruntime
  python export_models.py

產出：
  models/yolov8n.onnx       (~6.2 MB)  — 物件偵測
  models/midas_small.onnx   (~17 MB)   — 深度估測
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── YOLOv8n ───────────────────────────────
def export_yolo():
    out = os.path.join(MODELS_DIR, "yolov8n.onnx")
    if os.path.exists(out):
        print(f"[YOLO] 已存在: {out}")
        return

    print("[YOLO] 匯出 YOLOv8n → ONNX ...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, simplify=True, opset=17)
        # ultralytics exports to ./yolov8n.onnx
        src = "yolov8n.onnx"
        if os.path.exists(src):
            os.rename(src, out)
            print(f"[YOLO] ✓ 已匯出: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
        else:
            print("[YOLO] ✗ 匯出失敗，找不到輸出檔案")
    except ImportError:
        print("[YOLO] ✗ 請先安裝 ultralytics: pip install ultralytics")
        print("[YOLO] 嘗試從 HuggingFace 下載預先匯出的模型...")
        download_yolo_fallback(out)

def download_yolo_fallback(out):
    """從 HuggingFace 下載預先匯出的 YOLOv8n ONNX"""
    urls = [
        "https://huggingface.co/niconielsen32/yolov8n/resolve/main/yolov8n.onnx",
        "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx",
    ]
    for url in urls:
        try:
            print(f"[YOLO] 下載: {url}")
            urllib.request.urlretrieve(url, out, reporthook=progress_hook)
            print(f"\n[YOLO] ✓ 已下載: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"\n[YOLO] ✗ 下載失敗: {e}")
    print("[YOLO] ✗ 所有下載來源失敗，請手動匯出")

# ─── MiDaS Small ───────────────────────────
def download_midas():
    out = os.path.join(MODELS_DIR, "midas_small.onnx")
    if os.path.exists(out):
        print(f"[MiDaS] 已存在: {out}")
        return

    print("[MiDaS] 下載 MiDaS v2.1 Small ONNX ...")
    urls = [
        "https://huggingface.co/niconielsen32/midas-small/resolve/main/model_small.onnx",
        "https://github.com/niconielsen32/yolov8-onnxruntime-web/raw/main/public/midas_small.onnx",
    ]
    for url in urls:
        try:
            print(f"[MiDaS] 下載: {url}")
            urllib.request.urlretrieve(url, out, reporthook=progress_hook)
            print(f"\n[MiDaS] ✓ 已下載: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"\n[MiDaS] ✗ 下載失敗: {e}")

    # Fallback: export from torch hub
    try:
        print("[MiDaS] 嘗試從 PyTorch Hub 匯出 ...")
        import torch
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()
        dummy = torch.randn(1, 3, 256, 256)
        torch.onnx.export(model, dummy, out,
                          input_names=["input"],
                          output_names=["output"],
                          opset_version=17,
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"[MiDaS] ✓ 已匯出: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"[MiDaS] ✗ 匯出失敗: {e}")
        print("[MiDaS] 深度估測為選用功能，可以之後再安裝")

def progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 / total_size if total_size > 0 else 0
    print(f"\r  進度: {pct:.1f}%", end="", flush=True)

# ─── Main ──────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("A-Eye 模型匯出工具")
    print("=" * 50)
    export_yolo()
    print()
    download_midas()
    print()
    print("完成！請確認 models/ 目錄下有以下檔案：")
    for f in os.listdir(MODELS_DIR):
        fp = os.path.join(MODELS_DIR, f)
        print(f"  {f}  ({os.path.getsize(fp) / 1e6:.1f} MB)")
