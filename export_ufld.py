"""
Ultra-Fast-Lane-Detection (UFLD) → ONNX 匯出腳本

使用方式：
  1. git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git ufld
  2. 下載預訓練權重 tusimple_18.pth 放到 ufld/ 目錄
     （權重連結見 UFLD repo README）
  3. pip install torch torchvision onnx onnxruntime
  4. python export_ufld.py

會在 models/ 目錄產生 ufld_tusimple.onnx (~5MB)
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# ── 設定 ──
UFLD_REPO = "./ufld"                          # clone 路徑
WEIGHT_PATH = "./ufld/tusimple_18.pth"        # 預訓練權重
OUTPUT_PATH = "./models/ufld_tusimple.onnx"
INPUT_H, INPUT_W = 288, 800
NUM_GRIDDING = 100       # TuSimple: 100, CULane: 200
NUM_CLS = 56             # TuSimple row anchors
NUM_LANES = 4

# ── 加入 UFLD 到 path ──
sys.path.insert(0, UFLD_REPO)

try:
    from model.model import parsingNet
except ImportError:
    print("❌ 找不到 UFLD repo，請先執行：")
    print("   git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git ufld")
    sys.exit(1)

def export():
    print(f"[1/3] 載入模型 backbone=resnet18, griding={NUM_GRIDDING}, cls={NUM_CLS}")

    net = parsingNet(
        pretrained=False,
        backbone='18',
        cls_dim=(NUM_GRIDDING + 1, NUM_CLS, NUM_LANES),
        use_aux=False,
    )

    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到權重檔: {WEIGHT_PATH}")
        print("   請從 UFLD repo README 下載 tusimple_18.pth")
        sys.exit(1)

    state_dict = torch.load(WEIGHT_PATH, map_location='cpu')
    # 有些 checkpoint 會包在 'model' key 裡
    if 'model' in state_dict:
        state_dict = state_dict['model']
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    print("   ✅ 權重載入成功")

    print(f"[2/3] 匯出 ONNX → {OUTPUT_PATH}")
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    torch.onnx.export(
        net, dummy, OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,  # 固定 batch=1
    )
    print(f"   ✅ 匯出成功，檔案大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

    print("[3/3] 驗證 ONNX")
    import onnxruntime as ort
    sess = ort.InferenceSession(OUTPUT_PATH)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"   input:  {inp.name} {inp.shape}")
    print(f"   output: {out.name} {out.shape}")

    result = sess.run(None, {"input": np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)})
    print(f"   output shape: {result[0].shape}")
    print("   ✅ 驗證通過！")
    print(f"\n🎉 完成！請將 {OUTPUT_PATH} 部署到 A-Eye 的 models/ 目錄")

if __name__ == "__main__":
    export()
