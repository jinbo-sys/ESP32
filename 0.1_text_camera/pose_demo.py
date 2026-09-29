# -*- coding: utf-8 -*-
r"""
端侧模型姿态检测 Demo：用 TFLite 端侧模型实时检测人体关键点，画出
圆形人脸、躯干、手臂、手部（手腕），并显示"有无人脸"。按 q 退出。

运行方式：
  .\.venv\Scripts\Activate.ps1
  python pose_demo.py

【这就是"自己生成的端侧模型"的完整链路】
  1. 训练：在 PC 上用 TensorFlow 训练模型（本例用 MoveNet 姿态模型代替演示）
  2. 转换：tf.lite.TFLiteConverter 把模型转成 .tflite 文件（端侧格式）
  3. 推理：用 tf.lite.Interpreter 加载 .tflite 运行 —— 与 MCU 上 TFLM
     的 C++ API 是同一思想（加载 → 输入 tensor → invoke → 读输出 tensor）
  4. 换成自己训练的模型：只需把 models/ 下的 .tflite 替换掉，
     输入尺寸代码会自动读取，输出关键点顺序按你的模型定义调整即可。

【关键点说明（COCO 17 点，MoveNet 输出）】
  0 鼻  1/2 左右眼  3/4 左右耳  5/6 左右肩  7/8 左右肘  9/10 左右腕
  11/12 左右髋  13/14 左右膝  15/16 左右踝
  注意：MoveNet 只到"手腕"，指尖级检测需要手部专用模型（如 MediaPipe
  Hands 的 21 点模型，含每根手指的关节），本 demo 在手腕处画手部圆圈。
"""

import os
import time

import cv2
import numpy as np
import tensorflow as tf

# ========== 模型文件路径 ==========
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "movenet_lightning.tflite")

# 模型下载源：优先国内镜像，全部失败才报错
# MoveNet SinglePose Lightning int8 官方量化版（~2.9MB，17 关键点，输入 192x192 uint8）
MODEL_URLS = [
    "https://gh-proxy.com/https://raw.githubusercontent.com/Kazuhito00/MoveNet-Python-Example/main/tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite",
    "https://ghproxy.net/https://raw.githubusercontent.com/Kazuhito00/MoveNet-Python-Example/main/tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite",
    "https://cdn.jsdelivr.net/gh/Kazuhito00/MoveNet-Python-Example@main/tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite",
    "https://raw.githubusercontent.com/Kazuhito00/MoveNet-Python-Example/main/tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite",
]

# COCO 17 关键点索引
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL = 5, 6, 7, 8          # 左右肩、左右肘
LWR, RWR = 9, 10                          # 左右手腕
LHIP, RHIP = 11, 12                       # 左右髋


def download_file(urls, save_path, timeout=30):
    """依次尝试 urls 列表下载文件到 save_path，全部失败才抛异常。

    原理：国内直连 GitHub 常超时，用 gh-proxy / ghproxy 镜像加速；
    加浏览器 User-Agent 防拦截，30 秒超时防卡死。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"下载模型: {url}")
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(save_path, "wb") as f:
                    f.write(resp.read())
            print(f"已保存到: {save_path}")
            return
        except Exception as e:
            last_err = e
            print(f"  失败: {e}")
    raise RuntimeError(f"所有下载源均失败: {last_err}")


class TFLitePoseDetector:
    """端侧姿态检测器：加载 .tflite 模型，输出 17 个身体关键点。"""

    def __init__(self, score_threshold=0.3):
        """初始化：加载模型并读取输入/输出张量信息。

        原理：tf.lite.Interpreter 是 TFLite 的标准推理入口（端侧同款）。
        get_input_details/get_output_details 拿到输入输出张量的
        形状、数据类型、索引 —— 这些信息全部由模型文件自带，
        所以换自己的模型时输入尺寸无需改代码。
        """
        self.score_threshold = score_threshold
        if not os.path.exists(MODEL_PATH):
            download_file(MODEL_URLS, MODEL_PATH)
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()  # 分配张量内存（TFLM 里对应 AllocateTensors）
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_h = self.input_detail["shape"][1]  # 输入高度，如 192
        self.input_w = self.input_detail["shape"][2]  # 输入宽度，如 192
        print(f"端侧模型: {os.path.basename(MODEL_PATH)}")
        print(f"  输入: {self.input_detail['shape']} {self.input_detail['dtype']}")
        print(f"  输出: {self.output_detail['shape']} {self.output_detail['dtype']}")

    def detect(self, frame):
        """推理一帧，返回关键点数组 [17, 3]，每行 = (像素x, 像素y, 置信度)。

        原理（端侧推理四步曲）：
          1. 预处理：BGR→RGB，缩放到模型输入尺寸（如 192x192），
             按 dtype 转换（int8 模型要 uint8，float 模型要 float32）
          2. set_tensor：把数据写入输入张量
          3. invoke()：执行一次网络前向推理（对应 TFLM 的 Invoke）
          4. get_tensor：读输出张量 [1,1,17,3]，每点是 (y, x, score)，
             坐标为 0~1 归一化值，乘回原图宽高得到像素坐标
        """
        ih, iw = self.input_h, self.input_w
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (iw, ih))
        if self.input_detail["dtype"] == np.uint8:
            tensor = img.astype(np.uint8)
        else:
            tensor = img.astype(np.float32)  # MoveNet float 版输入范围是 0-255
        tensor = np.expand_dims(tensor, 0)   # 加 batch 维 → [1, ih, iw, 3]

        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])

        # 输出 [1,1,17,3] → [17,3]，每行 (y, x, score)，坐标是 0~1
        kpts = output[0, 0]  # [17, 3]
        h, w = frame.shape[:2]
        result = np.zeros((17, 3), dtype=np.float32)
        result[:, 0] = kpts[:, 1] * w       # x 像素坐标
        result[:, 1] = kpts[:, 0] * h       # y 像素坐标
        result[:, 2] = kpts[:, 2]           # 置信度
        return result


def _pt(kpts, i):
    """取第 i 个关键点的 (x, y)；置信度不足返回 None。"""
    if kpts[i, 2] > 0.3:
        return int(kpts[i, 0]), int(kpts[i, 1])
    return None


def _line(frame, p1, p2, color, thickness=3):
    """两点都有置信度时画线。"""
    if p1 and p2:
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_pose(frame, kpts):
    """画出圆形人脸、躯干、手臂、手部圆圈，返回 (是否有人脸, 是否有上身)。

    原理：
      - 人脸圆：圆心取鼻子，半径由两耳间距估计（正脸时耳距 ≈ 脸宽）；
        耳朵被遮挡时退化为两眼间距 × 1.6
      - 躯干：左右肩 + 左右髋 四点连线成四边形
      - 手臂：肩 → 肘 → 腕 两段折线（左右各一条）
      - 手部：手腕处画实心圆。指尖需手部专用模型（见文件头说明）
    """
    nose = _pt(kpts, NOSE)
    lear, rear = _pt(kpts, LEAR), _pt(kpts, REAR)
    leye, reye = _pt(kpts, LEYE), _pt(kpts, REYE)
    lsh, rsh = _pt(kpts, LSH), _pt(kpts, RSH)
    lel, rel = _pt(kpts, LEL), _pt(kpts, REL)
    lwr, rwr = _pt(kpts, LWR), _pt(kpts, RWR)
    lhip, rhip = _pt(kpts, LHIP), _pt(kpts, RHIP)

    face_present = nose is not None
    body_present = (lsh or rsh) is not None

    # ---- 1. 圆形人脸（青色）----
    if face_present:
        if lear and rear:
            radius = max(int(np.hypot(lear[0] - rear[0], lear[1] - rear[1])) // 2, 10)
        elif leye and reye:
            radius = max(int(np.hypot(leye[0] - reye[0], leye[1] - reye[1]) * 1.6), 10)
        else:
            radius = 20
        cv2.circle(frame, nose, radius, (255, 255, 0), 2, cv2.LINE_AA)

    # ---- 2. 躯干（黄色四边形）----
    _line(frame, lsh, rsh, (0, 255, 255))
    _line(frame, rsh, rhip, (0, 255, 255))
    _line(frame, rhip, lhip, (0, 255, 255))
    _line(frame, lhip, lsh, (0, 255, 255))

    # ---- 3. 手臂（绿色：左臂 / 洋红：右臂）----
    _line(frame, lsh, lel, (0, 255, 0))
    _line(frame, lel, lwr, (0, 255, 0))
    _line(frame, rsh, rel, (255, 0, 255))
    _line(frame, rel, rwr, (255, 0, 255))
    # 关节点画小圆，便于观察
    for p in (lsh, rsh, lel, rel):
        if p:
            cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)

    # ---- 4. 手部圆圈（手腕处，橙色实心）----
    # 指尖级检测需 MediaPipe Hands 21 点模型或自训手部模型，此处以手腕代手
    for p in (lwr, rwr):
        if p:
            cv2.circle(frame, p, 10, (0, 128, 255), -1, cv2.LINE_AA)

    return face_present, body_present


def main():
    """主流程：打开摄像头 → 加载端侧模型 → 循环推理画姿态 → 按 q 退出。"""
    # 摄像头打开逻辑与 camera_demo.py 相同（脚本同目录可直接导入复用）
    try:
        from camera_demo import open_camera
        cap = open_camera(0)
    except ImportError:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = TFLitePoseDetector(score_threshold=0.3)

    print("按 q 退出")
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败，退出")
            break
        frame = cv2.flip(frame, 1)  # 镜像，符合照镜子直觉

        kpts = detector.detect(frame)
        face_present, body_present = draw_pose(frame, kpts)

        # FPS 统计：端侧部署时这是核心指标（MCU 上就是推理毫秒数）
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0
        prev_time = cur_time

        cv2.putText(frame, "TFLite MoveNet (edge model)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 有无人脸 / 有人体的实时判断
        cv2.putText(frame, f"Face: {'Yes' if face_present else 'No'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if face_present else (0, 0, 255), 2)
        cv2.putText(frame, f"Body: {'Yes' if body_present else 'No'}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if body_present else (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("TFLite Pose Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("TFLite Pose Demo", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
