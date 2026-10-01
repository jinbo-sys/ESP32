# -*- coding: utf-8 -*-
"""
把自训练灰度头部检测模型转换为端侧 .tflite（uint8 输入 / float32 输出，int8 量化）。

量化校准数据使用 self_captured/ 自采集帧（含正/负样本），
预处理与训练、head_demo.py 完全一致：灰度读取 → 补边成正方形 →
缩放到 192×192 → 归一化 0~1，保证激活统计分布与真实推理一致。

运行:
  python convert_head.py
"""
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_KERAS = Path(__file__).parent / "models" / "head_model.keras"
MODEL_OUT = Path(__file__).parent / "models" / "head_model_int8.tflite"
SELF_DIR = Path(__file__).parent / "self_captured"
IMG_SIZE = 192


def load_padded_gray(path):
    """
    读取一帧并按推理管线预处理：灰度 → 边缘补边成正方形 → 缩放 192。

    与 head_demo.py 的摄像头帧预处理保持一致，避免直接 resize
    把 16:9 画面横向压扁，导致量化校准分布与推理分布不一致。
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[:2]
    side = max(W, H)
    top = (side - H) // 2
    left = (side - W) // 2
    padded = cv2.copyMakeBorder(
        img, top, side - H - top, left, side - W - left,
        cv2.BORDER_REPLICATE)
    return cv2.resize(padded, (IMG_SIZE, IMG_SIZE))


def representative_data_gen():
    """
    量化用代表性数据生成器（全部自采集帧，含无头负样本）。

    int8 全整数量化需要用真实输入统计每层激活值范围；
    训练时输入归一化到 0-1，这里必须保持一致，输出含 batch 维 (1,192,192,1)。
    """
    files = sorted(SELF_DIR.glob("*.jpg"))

    def _gen():
        for f in files:
            img = load_padded_gray(f).astype(np.float32) / 255.0
            yield [img[None, :, :, None]]

    return _gen


def main():
    """加载模型 → int8 量化转换 → 保存并做一次推理自检。"""
    if not MODEL_KERAS.exists():
        print(f"错误：找不到 {MODEL_KERAS}，请先运行 python train_self.py")
        return

    calib_files = sorted(SELF_DIR.glob("*.jpg"))
    if not calib_files:
        print(f"错误：{SELF_DIR} 下没有校准用自采集帧，请先运行 capture_self.py")
        return

    print(f"加载模型: {MODEL_KERAS}")
    model = tf.keras.models.load_model(MODEL_KERAS, compile=False)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    conv.representative_dataset = representative_data_gen()

    print(f"转换中（{len(calib_files)} 张自采集帧统计量化范围）...")
    data = conv.convert()
    MODEL_OUT.write_bytes(data)
    print(f"✓ 已生成 {MODEL_OUT}（{len(data)/1024:.1f} KB）")

    # 推理自检
    itp = tf.lite.Interpreter(model_path=str(MODEL_OUT))
    itp.allocate_tensors()
    ind, oud = itp.get_input_details()[0], itp.get_output_details()[0]
    print(f"输入 {ind['shape']} {ind['dtype']} → 输出 {oud['shape']} {oud['dtype']}")
    x = np.random.randint(0, 255, ind["shape"], np.uint8)
    itp.set_tensor(ind["index"], x)
    itp.invoke()
    y = itp.get_tensor(oud["index"])
    print(f"✓ 自检通过，输出范围 [{y.min():.3f}, {y.max():.3f}]")


if __name__ == "__main__":
    main()
